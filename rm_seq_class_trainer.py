import os
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"

import warnings
warnings.filterwarnings("ignore")

from unsloth import FastModel, FastLanguageModel, is_bfloat16_supported
from transformers import (
    AutoModelForSequenceClassification, AutoConfig, AutoTokenizer,
    TrainingArguments, Trainer, EvalPrediction,
    DataCollatorWithPadding)
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, cohen_kappa_score)

import torch
import pandas as pd
import numpy as np
import json

# ─────────────────────────────────────────────
# MODEL SELECTION DICTIONARY
# Uncomment exactly ONE entry to switch models.
# ─────────────────────────────────────────────

# ── Non-LLM Models ──────────────────────────
# MODEL_NAME = "microsoft/deberta-v3-base"        # DeBERTa-v3 Base
# MODEL_NAME = "microsoft/deberta-v3-large"       # DeBERTa-v3 Large
# MODEL_NAME = "unsloth/ModernBERT-large"         # ModernBERT Large
# MODEL_NAME = "EleutherAI/neoBERT-400M"          # NeoBERT 400M

# ── LLM Models (Qwen) ───────────────────────
# MODEL_NAME = "unsloth/Qwen3-0.6B"               # Qwen3 0.6B
# MODEL_NAME = "unsloth/Qwen3-4B"                 # Qwen3 4B
# MODEL_NAME = "unsloth/Qwen2.5-0.5B"             # Qwen2.5 0.5B
# MODEL_NAME = "unsloth/Qwen2.5-3B"               # Qwen2.5 3B

MODEL_NAME = "unsloth/ModernBERT-large"           # ← Active model

# ─────────────────────────────────────────────
# LABEL CONFIGURATION  (3-class)
# ─────────────────────────────────────────────
id2label     = {0: "NL", 1: "NM", 2: "NH"}
label2id     = {"NL": 0, "NM": 1, "NH": 2}
NUM_LABELS   = 3
TARGET_NAMES = ["NL", "NM", "NH"]

max_seq_length = 4096
COMPARISON_CSV = "outputs/model_comparison_report.csv"

# ─────────────────────────────────────────────
# STEP 1 — BUILD MODEL & TOKENIZER
# ─────────────────────────────────────────────

# LLM model names that should use FastLanguageModel + LoRA (Qwen etc.)
LLM_MODELS = (
    "qwen", "llama", "mistral", "gemma", "phi",
)

def _is_llm_model(model_name: str) -> bool:
    """Return True if the model is an LLM that should use FastLanguageModel."""
    return any(k in model_name.lower() for k in LLM_MODELS)


def build_model(model_name: str):
    """Load model for sequence classification using Unsloth where possible.

    Two paths:
    ─────────
    LLM (Qwen, LLaMA, etc.)
        Uses FastLanguageModel.from_pretrained() + get_peft_model() with LoRA.
        Unsloth provides 2× speed and 50% VRAM reduction.

    Encoder (ModernBERT, DeBERTa, NeoBERT)
        FastModel.from_pretrained() cannot set num_labels / problem_type on
        these models without a config conflict.  We load via
        AutoModelForSequenceClassification with a pre-built config so the
        classification head is created correctly (sequence-level, not
        token-level).  Unsloth kernel patches still apply at import time.
    """
    if _is_llm_model(model_name):
        # ── LLM path: Qwen, LLaMA, etc. ─────────────────────────────────
        print(f"  Loading via Unsloth FastLanguageModel (LLM path): {model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name      = model_name,
            max_seq_length  = max_seq_length,
            dtype           = None,
            load_in_4bit    = False,
        )
        # Add LoRA classification head
        model = FastLanguageModel.get_peft_model(
            model,
            r                          = 64,
            target_modules             = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha                 = 128,
            lora_dropout               = 0,
            bias                       = "none",
            use_gradient_checkpointing = "unsloth",
            random_state               = 3407,
            use_rslora                 = False,
            loftq_config               = None,
            task_type                  = "SEQ_CLS",
        )
        # Patch config for classification
        model.config.num_labels   = NUM_LABELS
        model.config.id2label     = id2label
        model.config.label2id     = label2id
        model.config.problem_type = "single_label_classification"

    else:
        # ── Encoder path: ModernBERT, DeBERTa, NeoBERT ──────────────────
        print(f"  Loading via AutoModelForSequenceClassification (encoder path): {model_name}")
        # Build config with all classification settings BEFORE model init
        # so the correct head size and problem_type are set from the start.
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels    = NUM_LABELS
        config.id2label      = id2label
        config.label2id      = label2id
        config.problem_type  = "single_label_classification"

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config                  = config,
            torch_dtype             = torch.bfloat16 if is_bfloat16_supported() else torch.float16,
            ignore_mismatched_sizes = True,
        )
        # Enable gradient checkpointing to reduce VRAM (same as Unsloth full_finetuning)
        model.gradient_checkpointing_enable()
        # Unsloth kernel patches still active from import-time monkey-patching

    # Verify classifier head output size
    import torch.nn as nn
    for attr_name in ("classifier", "score"):
        head = getattr(model, attr_name, None)
        if head is not None:
            out = (head.out_features if isinstance(head, nn.Linear)
                   else head.out_proj.out_features if hasattr(head, "out_proj")
                   else None)
            if out is not None:
                print(f"  Classifier head : {attr_name} → {out} outputs  ✓")
            break

    return model, tokenizer


# ─────────────────────────────────────────────
# STEP 2 — PROMPT BUILDER
# ─────────────────────────────────────────────
def create_messages(input_data: pd.Series) -> list:
    """
    Builds a chat-template-compatible message list for a single dataset row.

    Novice sub-level descriptors
    ─────────────────────────────
    NH  (Novice High)   — Communicates minimally on familiar topics using
                          isolated words, memorised phrases, or simple sentences.
                          Vocabulary slightly above lowest level; grammar errors
                          frequent; succeeds only in predictable exchanges.

    NM  (Novice Mid)    — Very limited inventory of isolated words and memorised
                          phrases. Short, fragmented, formulaic responses.
                          Comprehensibility depends heavily on listener goodwill.
                          Little evidence of productive grammar control.

    NL  (Novice Low)    — Only isolated words or echoed/copied fragments of the
                          prompt. No connected discourse; extremely limited in
                          quantity and quality. Pervasive errors often make the
                          response incomprehensible.
    """
    system_message = {
        "role": "system",
        "content": (
            "You are an expert ACTFL oral-proficiency rater specialising in "
            "Novice-level speech assessment.\n\n"
            "Your task is to assign one of three Novice sub-level ratings to "
            "a test-taker's spoken response based on the prompt and context "
            "provided.\n\n"
            "RATING DEFINITIONS\n"
            "──────────────────\n"
            "NH  (Novice High)\n"
            "    The speaker can communicate minimally on very familiar topics.\n"
            "    Responses use isolated words, memorised phrases, or simple "
            "sentences.  There is evidence of some vocabulary beyond the lowest "
            "level, but accuracy and range remain very limited.  Grammar errors "
            "are frequent; communication succeeds only in highly predictable "
            "exchanges.  The response, while limited, shows a slightly stronger "
            "command of language than NM.\n\n"
            "NM  (Novice Mid)\n"
            "    The speaker communicates using a very limited inventory of "
            "isolated words and memorised phrases.  Responses are short, "
            "fragmented, and heavily formulaic.  Comprehensibility depends "
            "greatly on the listener's goodwill and prior knowledge of context.  "
            "Little evidence of productive grammar control is present.\n\n"
            "NL  (Novice Low)\n"
            "    The speaker produces only isolated words or copied/echoed "
            "fragments of the prompt.  There is no connected discourse; utterances "
            "are extremely limited in both quantity and quality.  Errors are "
            "pervasive and often make the response incomprehensible.\n\n"
            "EVALUATION GUIDELINES\n"
            "─────────────────────\n"
            "• Focus on overall communicative quality, not isolated features.\n"
            "• Filler words, pauses, and natural disfluencies are expected at "
            "this level; treat them as evidence of spontaneity.\n"
            "• Consider whether the response directly addresses the prompt "
            "script or merely echoes / ignores it.\n"
            "• Weigh vocabulary range, grammatical control, and discourse "
            "coherence together.\n"
            "• Use the prompt level and test context to calibrate expectations."
        ),
    }

    user_prompt = (
        f"Prompt:          {input_data['prompt']}\n"
        f"Topic:           {input_data['topic']}\n"
        f"Prompt Script:   {input_data['prompt_script']}\n"
        f"Test:            {input_data['tests']}\n"
        f"Level:           {input_data['level']}\n"
        f"Test Taker ID:   {input_data['test_takers']}\n"
        f"Transcription:   {input_data['transcription']}\n\n"
        "Task:\n"
        "Predict the oral-proficiency sub-level rating of this test-taker's "
        "response.  Your answer must be exactly one of:\n"
        "  NH  (Novice High)\n"
        "  NM  (Novice Mid)\n"
        "  NL  (Novice Low)\n"
    )

    return [system_message, {"role": "user", "content": user_prompt}]


# ─────────────────────────────────────────────
# STEP 3 — DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
def load_and_preprocess_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read CSVs, rename columns, encode labels, and build message lists."""
    train_df = pd.read_csv("transcribed_3000_train.csv")
    test_df  = pd.read_csv("transcribed_3000_test.csv")

    print(f"  Train columns : {train_df.columns.tolist()}")
    print(f"  Test  columns : {test_df.columns.tolist()}")
    print(f"  Train shape   : {train_df.shape}")
    print(f"  Test  shape   : {test_df.shape}")

    # Map raw CSV header names → internal names used throughout the script.
    # Only renames columns that actually exist — safe to run even if some
    # headers are already correctly named or differ slightly.
    column_mapping = {
        # ── Actual CSV headers → internal names used throughout the script ──
        "Prompt":               "prompt",
        "Topic":                "topic",
        "Sub Topic":            "prompt_script",   # closest match for prompt_script
        "Form":                 "tests",
        "Prompt Level":         "level",
        "TestID":               "test_takers",
        "Machine Transcription":"transcription",
        "Final Rating":         "final_rating",
    }
    # Only rename columns that are actually present in the dataframe
    train_df.rename(columns={k: v for k, v in column_mapping.items()
                              if k in train_df.columns}, inplace=True)
    test_df.rename(columns={k: v for k, v in column_mapping.items()
                             if k in test_df.columns},  inplace=True)

    # Verify required columns exist after renaming
    required = ["prompt", "topic", "prompt_script", "tests",
                "level", "test_takers", "transcription", "final_rating"]
    missing_train = [c for c in required if c not in train_df.columns]
    missing_test  = [c for c in required if c not in test_df.columns]
    if missing_train or missing_test:
        raise ValueError(
            f"Missing columns after rename.\n"
            f"  train missing: {missing_train}\n"
            f"  test  missing: {missing_test}\n"
            f"  Actual train cols: {train_df.columns.tolist()}\n"
            "  → Update column_mapping in load_and_preprocess_data() to match."
        )

    train_df["label"] = train_df["final_rating"].map(label2id)
    test_df["label"]  = test_df["final_rating"].map(label2id)

    # Warn if any labels failed to map (NaN means an unseen label value)
    if train_df["label"].isna().any():
        bad = train_df.loc[train_df["label"].isna(), "final_rating"].unique()
        raise ValueError(f"Unknown label values in train set: {bad}. "
                         f"Expected one of: {list(label2id.keys())}")
    if test_df["label"].isna().any():
        bad = test_df.loc[test_df["label"].isna(), "final_rating"].unique()
        raise ValueError(f"Unknown label values in test set: {bad}. "
                         f"Expected one of: {list(label2id.keys())}")

    # ── Oversample minority classes in training set ─────────────────────
    # Upsample NL and NM rows to match NH count so the model sees balanced
    # classes without distorting the loss function with extreme weights.
    max_count = train_df["label"].value_counts().max()
    balanced_parts = []
    for lbl in sorted(train_df["label"].unique()):
        subset = train_df[train_df["label"] == lbl]
        upsampled = subset.sample(max_count, replace=True, random_state=42)
        balanced_parts.append(upsampled)
    train_df = pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)

    label_counts = train_df["label"].value_counts().sort_index()
    print(f"  Balanced train label counts: { {id2label[i]: label_counts[i] for i in sorted(label_counts.index)} }")

    train_df["messages"] = train_df.apply(create_messages, axis=1)
    test_df["messages"]  = test_df.apply(create_messages,  axis=1)

    return train_df, test_df


# ─────────────────────────────────────────────
# STEP 4 — TOKENISATION
# ─────────────────────────────────────────────
def _is_llm_tokenizer(tokenizer) -> bool:
    """Return True if the tokenizer supports chat templates (LLMs like Qwen).
    Encoder-only models (ModernBERT, DeBERTa, NeoBERT) have no chat_template.
    """
    return bool(getattr(tokenizer, "chat_template", None))


def _messages_to_plain_text(messages: list) -> str:
    """Flatten a chat message list into a single string for encoder tokenizers.
    Concatenates system + user content separated by a newline separator.
    """
    parts = []
    for msg in messages:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"[SYSTEM] {content}")
        elif role == "user":
            parts.append(f"[USER] {content}")
        else:
            parts.append(content)
    return "\n\n".join(parts)


def _msgs_to_text(msg_list: list) -> str:
    """Flatten a chat message list to a plain string for tokenisation."""
    text = ""
    for msg in msg_list:
        if msg["role"] == "system":
            text += msg["content"] + "\n\n"
        else:
            text += msg["content"]
    return text


def tokenise_datasets(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    tokenizer,
) -> tuple[Dataset, Dataset]:
    """Tokenise without padding — padding is done per-batch by DataCollatorWithPadding.
    This guarantees input_ids and labels always share the same batch dimension,
    avoiding the (48 vs 32) mismatch caused by per-map padding in DDP.
    """
    use_chat_template = _is_llm_tokenizer(tokenizer)
    print(f"  Tokenisation mode : {'chat_template (LLM)' if use_chat_template else 'plain tokenizer (encoder)'}")

    train_data = Dataset.from_pandas(
        train_df[["messages", "label", "level"]], preserve_index=False
    )
    test_data = Dataset.from_pandas(
        test_df[["messages", "label", "level"]], preserve_index=False
    )

    def tokenise(examples):
        # Convert messages to text
        texts = []
        for msg_list in examples["messages"]:
            text = ""
            for msg in msg_list:
                if msg["role"] == "system":
                    text += msg["content"] + "\n\n"
                else:
                    text += msg["content"]
            texts.append(text)
        # Tokenize WITHOUT return_tensors — Trainer handles tensor conversion
        tokenized = tokenizer(
            texts,
            padding    = True,
            truncation = True,
            max_length = max_seq_length,
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    remove_cols = [c for c in train_data.column_names if c != "level"]
    train_data = train_data.map(tokenise, batched=True, remove_columns=remove_cols)
    test_data  = test_data.map(tokenise,  batched=True, remove_columns=remove_cols)

    return train_data, test_data



# ─────────────────────────────────────────────
# STEP 5 — METRICS
# ─────────────────────────────────────────────
def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """Compute Accuracy, F1 (macro + weighted), QWK, confusion matrix, report."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc    = accuracy_score(labels, preds)
    f1_mac = f1_score(labels, preds, average="macro",    zero_division=0)
    f1_wt  = f1_score(labels, preds, average="weighted", zero_division=0)
    qwk    = cohen_kappa_score(labels, preds, weights="quadratic")
    conf   = confusion_matrix(labels, preds).tolist()
    report = classification_report(labels, preds, target_names=TARGET_NAMES)

    print("\n" + "="*60)
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  F1 Macro   : {f1_mac:.4f}  |  F1 Weighted : {f1_wt:.4f}")
    print(f"  QWK        : {qwk:.4f}")
    print("\n  Classification Report:\n", report)
    print("  Confusion Matrix:\n", np.array(conf))
    print("="*60 + "\n")

    return {
        "accuracy"             : acc,
        "f1_macro"             : f1_mac,
        "f1_weighted"          : f1_wt,
        "qwk"                  : qwk,
        "confusion_matrix"     : conf,
        "classification_report": report,
    }


# ─────────────────────────────────────────────
# STEP 6 — TRAINING
# ─────────────────────────────────────────────
def train_model(model, tokenizer, train_data, test_data, output_dir: str):
    """Configure TrainingArguments, build Trainer, and run training."""
    training_args = TrainingArguments(
        per_device_train_batch_size = 32,
        gradient_accumulation_steps = 2,
        per_device_eval_batch_size  = 64,
        warmup_steps                = 50,
        num_train_epochs            = 20,
        learning_rate               = 5e-5,
        fp16                        = not is_bfloat16_supported(),
        bf16                        = is_bfloat16_supported(),
        optim                       = "adamw_8bit",
        weight_decay                = 0.001,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        logging_strategy            = "epoch",
        lr_scheduler_type           = "linear",
        seed                        = 3407,
        output_dir                  = output_dir,
        report_to                   = "none",
        metric_for_best_model       = "qwk",
        greater_is_better           = True,
        auto_find_batch_size        = True,
        label_names                 = ["labels"],
    )

    trainer = Trainer(
        model            = model,
        processing_class = tokenizer,
        train_dataset    = train_data,
        eval_dataset     = test_data,
        compute_metrics  = compute_metrics,
        args             = training_args,
    )

    trainer_stats = trainer.train()
    return trainer, trainer_stats


# ─────────────────────────────────────────────
# STEP 7 — SAVE MODEL & STATS
# ─────────────────────────────────────────────
def save_model_and_stats(trainer, tokenizer, trainer_stats, output_dir: str):
    """Persist the fine-tuned model, tokenizer, and training stats to disk."""
    trainer.save_model(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")

    with open(f"{output_dir}/trainer_stats.json", "w") as f:
        json.dump(str(trainer_stats), f)   # TrainOutput is not JSON-serialisable by default

    print(f"\n  Model saved → {output_dir}/final_model")


# ─────────────────────────────────────────────
# STEP 8 — PER-PROMPT-LEVEL ANALYTICS
# ─────────────────────────────────────────────
def per_prompt_level_analytics(
    trainer, test_data: Dataset, test_df: pd.DataFrame, output_dir: str
) -> pd.DataFrame:
    """
    Run inference on the test set and report F1 / QWK / Accuracy broken down
    by prompt level.  Saves a CSV summary to output_dir.
    """
    predictions = trainer.predict(test_data)
    preds  = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    df_results = test_df[["level"]].copy().reset_index(drop=True)
    df_results["pred"]  = preds
    df_results["label"] = labels

    print("\n" + "="*60)
    print("  PER-PROMPT-LEVEL ANALYTICS")
    print("="*60)

    summary_rows = []
    for lvl, grp in df_results.groupby("level"):
        y_true = grp["label"].values
        y_pred = grp["pred"].values
        n      = len(grp)
        acc    = accuracy_score(y_true, y_pred)
        f1     = f1_score(y_true, y_pred, average="macro", zero_division=0)
        qwk    = (
            cohen_kappa_score(y_true, y_pred, weights="quadratic")
            if len(np.unique(y_true)) > 1 else float("nan")
        )
        summary_rows.append({
            "Level": lvl, "N": n,
            "Accuracy": round(acc, 4),
            "F1_Macro": round(f1,  4),
            "QWK":      round(qwk, 4),
        })
        print(f"\n  Level: {lvl}  (n={n})")
        print(f"    Accuracy={acc:.4f}  F1_Macro={f1:.4f}  QWK={qwk:.4f}")
        print(classification_report(y_true, y_pred, target_names=TARGET_NAMES))
        print(confusion_matrix(y_true, y_pred))

    summary_df = pd.DataFrame(summary_rows)
    print("\n  Summary Table:\n", summary_df.to_string(index=False))
    summary_df.to_csv(f"{output_dir}/per_level_metrics.csv", index=False)
    return summary_df


# ─────────────────────────────────────────────
# STEP 9 — MODEL COMPARISON REPORT
# ─────────────────────────────────────────────
def update_comparison_report(trainer, model_name: str):
    """
    Evaluate the trained model and append / update one row in the shared
    model comparison CSV so all experiment runs can be compared at a glance.
    """
    os.makedirs("outputs", exist_ok=True)
    final_metrics = trainer.evaluate()

    row = {
        "Model"       : model_name,
        "Accuracy"    : round(final_metrics.get("eval_accuracy",    float("nan")), 4),
        "F1_Macro"    : round(final_metrics.get("eval_f1_macro",    float("nan")), 4),
        "F1_Weighted" : round(final_metrics.get("eval_f1_weighted", float("nan")), 4),
        "QWK"         : round(final_metrics.get("eval_qwk",         float("nan")), 4),
    }

    if os.path.exists(COMPARISON_CSV):
        comp_df = pd.read_csv(COMPARISON_CSV)
        existing = comp_df[comp_df["Model"] == model_name]
        if not existing.empty:
            prev_qwk = existing["QWK"].values[0]
            if row["QWK"] > prev_qwk:
                print(f"  New run QWK ({row['QWK']}) > previous ({prev_qwk}) — updating.")
                comp_df = comp_df[comp_df["Model"] != model_name]
                comp_df = pd.concat([comp_df, pd.DataFrame([row])], ignore_index=True)
            else:
                print(f"  New run QWK ({row['QWK']}) <= previous ({prev_qwk}) — keeping previous best.")
        else:
            comp_df = pd.concat([comp_df, pd.DataFrame([row])], ignore_index=True)
    else:
        comp_df = pd.DataFrame([row])

    comp_df.to_csv(COMPARISON_CSV, index=False)

    print("\n" + "="*60)
    print("  MODEL COMPARISON TABLE (all completed runs)")
    print("="*60)
    print(comp_df.to_string(index=False))
    print("="*60)



# ─────────────────────────────────────────────
# MAIN  — orchestrates all steps in order
# ─────────────────────────────────────────────
def main():
    # Derive isolated output directory from model name
    model_slug = MODEL_NAME.replace("/", "_")
    output_dir = f"outputs/{model_slug}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Starting experiment : {MODEL_NAME}")
    print(f"  Output dir          : {output_dir}")
    print(f"{'='*60}\n")

    # Step 1 — load model & tokenizer
    model, tokenizer = build_model(MODEL_NAME)

    # Step 2 + 3 — load data, rename columns, encode labels, build prompts
    train_df, test_df = load_and_preprocess_data()

    # Step 4 — tokenise with Qwen chat template
    train_data, test_data = tokenise_datasets(train_df, test_df, tokenizer)

    # Step 5 + 6 — train  (compute_metrics called internally by Trainer)
    trainer, trainer_stats = train_model(
        model, tokenizer, train_data, test_data, output_dir
    )

    # Step 7 — save model, tokenizer, and training stats
    save_model_and_stats(trainer, tokenizer, trainer_stats, output_dir)

    # Step 8 — per-prompt-level analytics on test set
    per_prompt_level_analytics(trainer, test_data, test_df, output_dir)

    # Step 9 — update global model comparison report
    update_comparison_report(trainer, MODEL_NAME)

    print("\nFinished Training")


if __name__ == "__main__":
    main()