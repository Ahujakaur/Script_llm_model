import inspect
import os
import json
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

warnings.filterwarnings("ignore")
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"

LABELS = ["NH", "NM", "NL"]
LABEL2ID: Dict[str, int] = {k: i for i, k in enumerate(LABELS)}
ID2LABEL: Dict[int, str] = {i: k for k, i in LABEL2ID.items()}
TEXT_COL = "Machine Transcription"
PROMPT_COL = "Prompt"
PROMPT_LEVEL_COL = "Prompt Level"
TARGET_COL = "Final Rating"
ID_COL = "PromptID"

DEFAULT_MODELS: List[str] = [
    "microsoft/deberta-v3-base",
    "unsloth/ModernBERT-large",
    "EleutherAI/neoBERT-400M",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen1.5-4B-Chat",
]

MAX_SEQ_LENGTH = 2048


def load_data(train_path: str, test_path: str) -> DatasetDict:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df[[ID_COL, PROMPT_COL, PROMPT_LEVEL_COL, TEXT_COL, TARGET_COL]].dropna()
        df["text"] = (
            "Prompt Level: " + df[PROMPT_LEVEL_COL].astype(str)
            + "\nPrompt: " + df[PROMPT_COL].astype(str)
            + "\nResponse: " + df[TEXT_COL].astype(str)
            + "\nRate as: NH (High), NM (Medium), NL (Low)"
        )
        df["label"] = df[TARGET_COL].map(LABEL2ID)
        return df[[ID_COL, PROMPT_LEVEL_COL, "text", "label"]]

    train_df = _prepare(train_df)
    test_df = _prepare(test_df)

    return DatasetDict(
        train=Dataset.from_pandas(train_df, preserve_index=False),
        test=Dataset.from_pandas(test_df, preserve_index=False),
    )


def tokenize_function(tokenizer, examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=MAX_SEQ_LENGTH,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro")
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=LABELS, output_dict=True)
    return {
        "macro_f1": macro_f1,
        "qwk": qwk,
        "confusion_matrix": cm.tolist(),
        "cls_report": report,
    }


def per_prompt_metrics(dataset: Dataset, preds: np.ndarray, labels: np.ndarray) -> Dict[str, Dict]:
    df = pd.DataFrame(dataset)[[ID_COL, PROMPT_LEVEL_COL]]
    df["pred"] = preds
    df["label"] = labels
    grouped = {}
    for pid, group in df.groupby(ID_COL):
        gpred, glabel = group["pred"].to_numpy(), group["label"].to_numpy()
        grouped[str(pid)] = {
            "macro_f1": f1_score(glabel, gpred, average="macro"),
            "qwk": cohen_kappa_score(glabel, gpred, weights="quadratic"),
            "support": len(group),
            "confusion_matrix": confusion_matrix(glabel, gpred).tolist(),
            "cls_report": classification_report(glabel, gpred, target_names=LABELS, output_dict=True),
            "prompt_levels": group[PROMPT_LEVEL_COL].unique().tolist(),
        }
    return grouped


def plot_confusion_matrix(cm: np.ndarray, model_name: str, save_path: str):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_class_distribution(labels: np.ndarray, save_path: str):
    unique, counts = np.unique(labels, return_counts=True)
    class_names = [LABELS[i] for i in unique]
    plt.figure(figsize=(8, 5))
    plt.bar(class_names, counts, color=["#2ecc71", "#f39c12", "#e74c3c"])
    plt.title("Class Distribution")
    plt.ylabel("Count")
    plt.xlabel("Rating Class")
    for i, v in enumerate(counts):
        plt.text(i, v + 20, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_curves(train_history: Dict, save_path: str):
    if not train_history:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if "loss" in train_history:
        axes[0].plot(train_history["loss"], label="Training Loss", marker="o")
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    if "eval_loss" in train_history:
        axes[1].plot(train_history["eval_loss"], label="Validation Loss", marker="s", color="orange")
        axes[1].set_title("Validation Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_experiment(
    model_name: str,
    train_path: str,
    test_path: str,
    output_dir: str = "outputs",
    num_epochs: int = 3,
    lr: float = 3e-5,
    batch_size: int = 8,
    grad_accum: int = 1,
    warmup_steps: int = 100,
    save_total_limit: int = 2,
):
    data = load_data(train_path, test_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    encoded = data.map(
        lambda x: tokenize_function(tokenizer, x),
        batched=True,
        remove_columns=["text"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    args_kwargs = dict(
        output_dir=os.path.join(output_dir, model_name.replace("/", "_")),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit=save_total_limit,
    )

    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        args_kwargs.update(
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
        )
    elif "eval_strategy" in sig.parameters:
        args_kwargs.update(
            eval_strategy="epoch",
            save_strategy="epoch" if "save_strategy" in sig.parameters else args_kwargs.get("save_strategy", "epoch"),
            logging_strategy="epoch" if "logging_strategy" in sig.parameters else args_kwargs.get("logging_strategy", "epoch"),
        )

    args = TrainingArguments(**args_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["test"],
        compute_metrics=compute_metrics,
    )

    trainer_sig = inspect.signature(Trainer.__init__)
    if "data_collator" in trainer_sig.parameters:
        trainer_kwargs["data_collator"] = data_collator
    if "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    logits, labels, _ = trainer.predict(encoded["test"], metric_key_prefix="test")
    preds = np.argmax(logits, axis=-1)
    prompt_metrics = per_prompt_metrics(encoded["test"], preds, labels)

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(save_dir, "eval_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2)
    with open(os.path.join(save_dir, "prompt_metrics.json"), "w") as f:
        json.dump(prompt_metrics, f, indent=2)
    with open(os.path.join(save_dir, "train_result.json"), "w") as f:
        json.dump(train_result.metrics, f, indent=2)
    
  
    cm = np.array(eval_metrics["confusion_matrix"])
    plot_confusion_matrix(cm, model_name.split("/")[-1], os.path.join(save_dir, "confusion_matrix.png"))
    plot_class_distribution(labels, os.path.join(save_dir, "class_distribution.png"))
    plot_training_curves(train_result.metrics, os.path.join(save_dir, "training_curves.png"))

    return {
        "model": model_name,
        "macro_f1": eval_metrics.get("macro_f1", 0),
        "qwk": eval_metrics.get("qwk", 0),
        "eval": eval_metrics,
        "prompt_metrics": prompt_metrics,
    }


def plot_model_comparison(results: List[Dict], output_dir: str):
    models = [r["model"].split("/")[-1] for r in results]
    f1_scores = [r["macro_f1"] for r in results]
    qwk_scores = [r["qwk"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].barh(models, f1_scores, color="#3498db")
    axes[0].set_xlabel("Macro F1 Score")
    axes[0].set_title("Model Comparison - F1 Score")
    axes[0].set_xlim([0, 1])
    for i, v in enumerate(f1_scores):
        axes[0].text(v + 0.02, i, f"{v:.3f}", va="center")
    
    axes[1].barh(models, qwk_scores, color="#e74c3c")
    axes[1].set_xlabel("Quadratic Weighted Kappa")
    axes[1].set_title("Model Comparison - QWK")
    axes[1].set_xlim([0, 1])
    for i, v in enumerate(qwk_scores):
        axes[1].text(v + 0.02, i, f"{v:.3f}", va="center")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


def run_all(
    train_path: str = "transcribed_3000_train .csv",
    test_path: str = "transcribed_3000_test .csv",
    models: List[str] = None,
    output_dir: str = "outputs",
) -> List[Dict]:
    if models is None:
        models = DEFAULT_MODELS
    
    results = []
    for m in models:
        try:
            print(f"\n{'='*60}")
            print(f"Training: {m}")
            print(f"{'='*60}")
            res = run_experiment(
                model_name=m,
                train_path=train_path,
                test_path=test_path,
                output_dir=output_dir,
            )
            results.append(res)
            print(f"✓ F1: {res['macro_f1']:.4f} | QWK: {res['qwk']:.4f}")
        except Exception as e:
            print(f"✗ Error training {m}: {str(e)}")
            continue
    
    os.makedirs(output_dir, exist_ok=True)
    

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    
  
    if results:
        plot_model_comparison(results, output_dir)
        print(f"\n✓ Model comparison plot saved to {output_dir}/model_comparison.png")
    
    return results


if __name__ == "__main__":
    run_all()
