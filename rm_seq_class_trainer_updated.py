import os
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix, classification_report
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from unsloth import FastLanguageModel

warnings.filterwarnings("ignore")

# -----------------------
# Configuration
# -----------------------
MODEL_CANDIDATES: Dict[str, str] = {
    "qwen3_0_5b": "unsloth/Qwen3-0.5B",
    "qwen3_1_8b": "unsloth/Qwen3-1.8B",
    "qwen3_4b": "unsloth/Qwen3-4B",
    "qwen3_5_0_5b": "unsloth/Qwen3.5-0.5B",
    "qwen3_5_1_8b": "unsloth/Qwen3.5-1.8B",
    "qwen3_5_4b": "unsloth/Qwen3.5-4B",
}

MODEL_KEY = "qwen3_5_0_5b"
MODEL_NAME = MODEL_CANDIDATES[MODEL_KEY]

TRAIN_PATH = "data/novice_train.csv"
TEST_PATH = "data/novice_val.csv"
MAX_LENGTH = 4096
SEED = 3407

# Label mapping: NH = Not Harmful (low risk), NM = Moderately Harmful (medium risk), NL = Least Harmful (high quality/safe)
label2id: Dict[str, int] = {"NH": 0, "NM": 1, "NL": 2}
id2label: Dict[int, str] = {v: k for k, v in label2id.items()}


def create_messages(row: pd.Series) -> str:
    """
    Create verbose instruction-style prompt for LLM-based text classification.
    
    This function formats the input data into a comprehensive prompt that includes:
    - System instructions for the classification task
    - All relevant metadata (topic, prompt script, test level, test takers)
    - The actual prompt and transcription to be classified
    
    Args:
        row: DataFrame row containing all columns
    
    Returns:
        Formatted message string for classification
    """
    system_instruction = """You are an expert text classification system designed to evaluate spoken language proficiency and content quality.

Your task is to classify the given spoken response into one of the following three rating categories:

NH (Not Harmful): Response demonstrates low proficiency, contains significant issues, or is inappropriate.
NM (Moderately Harmful): Response shows moderate proficiency with some issues but acceptable overall quality.
NL (Least Harmful): Response demonstrates high proficiency, excellent quality, and is completely appropriate.

Carefully analyze the prompt, context, and transcribed response to determine the appropriate rating."""

    # Extract all relevant fields
    topic = row.get("Topic", "")
    prompt_script = row.get("Prompt", "")
    test_level = row.get("Test Level RM", "")
    test_takers = row.get("Prompt Level", "")
    transcription = row.get("Machine Transcription", "")

    user_text = f"""Context Information:
- Topic: {topic}
- Test Level: {test_level}
- Test Takers Level: {test_takers}

Prompt: {prompt_script}

Transcribed Response: {transcription}

Based on the above information, classify this response with the appropriate rating (NH, NM, or NL)."""

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
    df = df.dropna(subset=["Final Rating"])
    df = df[df["Final Rating"].isin(label2id.keys())].copy()

    # Fill NaN values for all columns used in create_messages
    df["Topic"] = df["Topic"].fillna("")
    df["Prompt"] = df["Prompt"].fillna("")
    df["Test Level RM"] = df["Test Level RM"].fillna("")
    df["Prompt Level"] = df["Prompt Level"].fillna("")
    df["Machine Transcription"] = df["Machine Transcription"].fillna("")
    
    # Use create_messages with all columns
    df["text"] = df.apply(create_messages, axis=1)
    df["label"] = df["Final Rating"].map(label2id)
    return df[["text", "label"]]


def tokenize_function(tokenizer, examples):
    # Apply Qwen chat template
    messages_list = [[{"role": "user", "content": text}] for text in examples["text"]]
    formatted_texts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) for msgs in messages_list]
    
    return tokenizer(
        formatted_texts,
        truncation=True,
        max_length=MAX_LENGTH,
    )


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    cm = confusion_matrix(labels, preds)
    print("\n" + "="*50)
    print("CONFUSION MATRIX:")
    print("="*50)
    print(f"Labels: {list(id2label.values())}")
    print(cm)
    print("="*50 + "\n")
    
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

    # Load model and tokenizer using Unsloth for 50% less memory and 2x faster training
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_LENGTH,
        dtype=get_torch_dtype(),
        load_in_4bit=True,
    )
    
    # Configure model for sequence classification
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"

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

    torch_dtype = get_torch_dtype()
    training_args = TrainingArguments(
        output_dir=f"outputs/{MODEL_KEY}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=5,
        learning_rate=2e-4,
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
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
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
