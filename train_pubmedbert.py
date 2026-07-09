"""
STEP 2 - Fine-tune PubMedBERT as an ICD-10 classifier.

Loads the prepared train/val splits, fine-tunes PubMedBERT for sequence
classification, and saves the best model + tokenizer to ./model.

Run (after prepare_data.py):
    python train_pubmedbert.py

Fast end-to-end test (1 epoch on a small subset, no GPU needed):
    python train_pubmedbert.py --smoke
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import config


def load_splits():
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    return train_df, val_df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    # top-5 accuracy: is the true label among the 5 highest-scoring codes?
    top5 = np.argsort(logits, axis=-1)[:, -5:]
    top5_acc = np.mean([lab in row for lab, row in zip(labels, top5)])
    return {
        "accuracy": accuracy_score(labels, preds),
        "top5_accuracy": top5_acc,
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def build_training_args(num_epochs: int) -> TrainingArguments:
    """Build TrainingArguments in a way that works across transformers versions.

    Newer versions renamed `evaluation_strategy` -> `eval_strategy`; older ones
    only accept `evaluation_strategy`. We detect which keyword is supported.
    """
    common = dict(
        output_dir=os.path.join(config.MODEL_DIR, "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=config.RANDOM_SEED,
    )
    try:
        return TrainingArguments(eval_strategy="epoch", **common)
    except TypeError:
        # Older transformers releases use the long name.
        return TrainingArguments(evaluation_strategy="epoch", **common)


def main(smoke: bool = False) -> None:
    if not (os.path.exists(config.TRAIN_CSV) and os.path.exists(config.VAL_CSV)):
        raise FileNotFoundError(
            "Prepared splits not found. Run `python prepare_data.py` first."
        )

    with open(config.LABEL_MAP_PATH, encoding="utf-8") as fh:
        maps = json.load(fh)
    label2id = {k: int(v) for k, v in maps["label2id"].items()}
    id2label = {int(k): v for k, v in maps["id2label"].items()}
    num_labels = len(label2id)
    print(f"Loaded {num_labels} ICD-10 classes.")

    train_df, val_df = load_splits()
    num_epochs = config.NUM_EPOCHS
    if smoke:
        # Fast pipeline check: tiny subset, single epoch.
        train_df = train_df.sample(n=min(1000, len(train_df)), random_state=config.RANDOM_SEED)
        val_df = val_df.sample(n=min(300, len(val_df)), random_state=config.RANDOM_SEED)
        num_epochs = 1
        print("[SMOKE TEST] Using reduced data and 1 epoch.")
    print(f"Train rows: {len(train_df):,} | Val rows: {len(val_df):,}")

    print(f"Loading base model: {config.BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_MODEL,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.MAX_LENGTH,
        )

    train_ds = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False)
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = build_training_args(num_epochs)

    # `Trainer` renamed the `tokenizer` argument to `processing_class` in
    # transformers >= 4.46. Pick whichever the installed version supports.
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    try:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)

    print("\n=== Training ===")
    trainer.train()

    print("\n=== Final validation metrics ===")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\nSaving best model -> {config.MODEL_DIR}")
    trainer.save_model(config.MODEL_DIR)
    tokenizer.save_pretrained(config.MODEL_DIR)

    print("\n✅ Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast end-to-end test: 1 epoch on a small subset.",
    )
    args = parser.parse_args()
    main(smoke=args.smoke)
