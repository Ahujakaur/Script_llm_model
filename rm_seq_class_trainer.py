import os
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix, classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

# -----------------------
# Configuration
# -----------------------
MODEL_CANDIDATES: Dict[str, str] = {
    "deberta_v3_base": "microsoft/deberta-v3-base",
    "modernbert_large": "unsloth/ModernBERT-large",
    "neobert_400m": "EleutherAI/neoBERT-400M",
    "qwen3_0_5b": "Qwen/Qwen3-0.5B",
    "qwen3_1_8b": "Qwen/Qwen3-1.8B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_5_0_5b": "Qwen/Qwen3.5-0.5B",
    "qwen3_5_1_8b": "Qwen/Qwen3.5-1.8B",
    "qwen3_5_4b": "Qwen/Qwen3.5-4B",
}

MODEL_KEY = "deberta_v3_base"  # change to sweep between candidates above
MODEL_NAME = MODEL_CANDIDATES[MODEL_KEY]

TRAIN_PATH = "transcribed_3000_train (1).csv"
TEST_PATH = "transcribed_3000_test (1).csv"
MAX_LENGTH = 512
SEED = 3407

# Label mapping for Final Rating (NH = high, NM = medium, NL = low)
label2id: Dict[str, int] = {"NH": 0, "NM": 1, "NL": 2}
id2label: Dict[int, str] = {v: k for k, v in label2id.items()}


def create_messages(prompt: str, transcription: str) -> str:
    """
    Create verbose instruction-style prompt for LLM-based text classification.
    
    Args:
        prompt: The original prompt text
        transcription: The machine transcription text
    
    Returns:
        Formatted message string for classification
    """
    system_instruction = """You are a text classification system designed to evaluate content safety.

Your task is to classify the given text into one of the following three categories:

NH - Non Harmful: Content that is safe, appropriate, and does not contain harmful elements.
NM - Mildly Harmful: Content that contains minor issues or potentially problematic elements but is not severely harmful.
NL - Highly Harmful: Content that is dangerous, inappropriate, or contains severely harmful elements.

Read the text carefully and output ONLY the label (NH, NM, or NL)."""

    user_text = f"""Prompt: {prompt}

Response: {transcription}

Based on the above prompt and response, classify the content."""

    # Format as instruction-style message
    formatted_message = f"{system_instruction}\n\n{user_text}"
    
    return formatted_message


def get_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_and_prepare_split(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["Final Rating", "Machine Transcription"])
    df = df[df["Final Rating"].isin(label2id.keys())].copy()

    # Use create_messages to format text with verbose prompt
    df["Prompt"] = df["Prompt"].fillna("")
    df["Machine Transcription"] = df["Machine Transcription"].fillna("")
    df["text"] = df.apply(
        lambda row: create_messages(row["Prompt"], row["Machine Transcription"]),
        axis=1
    )
    df["label"] = df["Final Rating"].map(label2id)
    return df[["text", "label"]]


def tokenize_function(tokenizer, examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    # Compute confusion matrix and save it
    cm = confusion_matrix(labels, preds)
    print("\n" + "="*50)
    print("CONFUSION MATRIX:")
    print("="*50)
    print(f"Labels: {list(id2label.values())}")
    print(cm)
    print("="*50 + "\n")
    
    # Compute classification report
    report = classification_report(labels, preds, target_names=list(id2label.values()), digits=4)
    print("CLASSIFICATION REPORT:")
    print(report)
    
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "qwk": cohen_kappa_score(labels, preds, weights="quadratic"),
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_df = load_and_prepare_split(TRAIN_PATH)
    test_df = load_and_prepare_split(TEST_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"

    torch_dtype = get_torch_dtype()
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch_dtype,
    )

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    train_dataset = train_dataset.map(
        lambda batch: tokenize_function(tokenizer, batch),
        batched=True,
        remove_columns=["text"],
    )
    test_dataset = test_dataset.map(
        lambda batch: tokenize_function(tokenizer, batch),
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=f"outputs/{MODEL_KEY}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=torch_dtype == torch.float16,
        bf16=torch_dtype == torch.bfloat16,
        lr_scheduler_type="linear",
        metric_for_best_model="macro_f1",
        load_best_model_at_end=True,
        greater_is_better=True,
        seed=SEED,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer_stats = trainer.train()

    trainer.save_model(f"outputs/{MODEL_KEY}/final_model")
    tokenizer.save_pretrained(f"outputs/{MODEL_KEY}/final_model")

    with open(os.path.join(training_args.output_dir, "trainer_stats.json"), "w") as f:
        f.write(trainer_stats.to_json_string())

    # Final evaluation with detailed metrics
    print("\n" + "="*50)
    print("FINAL EVALUATION ON TEST SET")
    print("="*50)
    eval_results = trainer.evaluate()
    print(f"\nFinal Metrics:")
    print(f"  Accuracy: {eval_results.get('eval_accuracy', 0):.4f}")
    print(f"  Macro F1: {eval_results.get('eval_macro_f1', 0):.4f}")
    print(f"  QWK: {eval_results.get('eval_qwk', 0):.4f}")
    print("="*50 + "\n")

    print("Finished training", MODEL_NAME)


if __name__ == "__main__":
    main()
