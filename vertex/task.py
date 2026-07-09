"""
Vertex AI custom training entrypoint - fine-tune PubMedBERT as an ICD-10 classifier.

Designed to run inside a GPU container on Vertex AI Custom Training.

Key differences from the local train_pubmedbert.py:
  * All paths are passed as CLI args and may be gs:// URIs (read via gcsfs).
  * The trained model is written to AIP_MODEL_DIR (Vertex auto-uploads it to GCS
    and registers it), or to --model_dir if provided.
  * Optionally reports the eval accuracy to the hyperparameter tuning service.

Two data-input modes:
  1. Spreadsheet mode (preprocess in the cloud): pass --excel_path gs://...xlsx
     and the column/split options. Cleaning + splitting happen here.
  2. CSV mode (preprocess locally): pass --train_path/--val_path/--label_map_path.

Example (spreadsheet mode):
    python task.py \
        --excel_path gs://icdpredictor123/train.csv \
        --sheet_name Sheet1 --col_source Source \
        --col_diagnosis Diagnosis --col_icd ICD \
        --val_size 3000 --drop_duplicates --min_samples_per_class 2
"""
import argparse
import json
import os
import re
import tempfile

import fsspec
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# GCS-aware helpers
# ---------------------------------------------------------------------------
def read_json(path: str) -> dict:
    with fsspec.open(path, "r") as fh:
        return json.load(fh)


def read_csv(path: str) -> pd.DataFrame:
    # pandas reads gs:// transparently when gcsfs is installed.
    return pd.read_csv(path)


def read_table(path: str, sheet_name=0) -> pd.DataFrame:
    """Read a table from GCS/local, auto-detecting the real format.

    Sniffs the leading magic bytes so a file is parsed by its true content
    regardless of its extension (e.g. a CSV mistakenly renamed to .xlsx):
      * b"PK\\x03\\x04"          -> modern .xlsx (openpyxl)
      * b"\\xd0\\xcf\\x11\\xe0"  -> legacy .xls
      * anything else            -> CSV
    """
    with fsspec.open(path, "rb") as fh:
        head = fh.read(8)
    if head[:4] == b"PK\x03\x04":
        with fsspec.open(path, "rb") as fh:
            return pd.read_excel(fh, sheet_name=sheet_name, engine="openpyxl")
    if head[:4] == b"\xd0\xcf\x11\xe0":
        with fsspec.open(path, "rb") as fh:
            return pd.read_excel(fh, sheet_name=sheet_name)
    return pd.read_csv(path)


def write_json(obj: dict, path: str) -> None:
    with fsspec.open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def write_csv(df: pd.DataFrame, path: str) -> None:
    with fsspec.open(path, "w", newline="") as fh:
        df.to_csv(fh, index=False)


def upload_dir(local_dir: str, dest: str) -> None:
    """Copy a local directory to dest, which may be a gs:// URI or a local path."""
    if dest.startswith("gs://"):
        fs = fsspec.filesystem("gcs")
        for root, _, files in os.walk(local_dir):
            for name in files:
                local_path = os.path.join(root, name)
                rel = os.path.relpath(local_path, local_dir)
                remote_path = dest.rstrip("/") + "/" + rel.replace(os.sep, "/")
                fs.put_file(local_path, remote_path)
    else:
        import shutil

        os.makedirs(dest, exist_ok=True)
        for root, _, files in os.walk(local_dir):
            for name in files:
                local_path = os.path.join(root, name)
                rel = os.path.relpath(local_path, local_dir)
                out_path = os.path.join(dest, rel)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                shutil.copy2(local_path, out_path)


# ---------------------------------------------------------------------------
# Preprocessing (spreadsheet mode) - mirrors the local prepare_data.py
# ---------------------------------------------------------------------------
def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().strip(":").strip()


def clean_icd(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).upper().replace(".", "").replace(" ", "").strip()


def build_input_text(source: str, diagnosis: str) -> str:
    # NOTE: must match the serving predictor and the original training format.
    return f"Specimen Source: {source} [SEP] Final Diagnosis: {diagnosis}"


def ensure_text_column(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Guarantee a string `text` column for prepared splits.

    prepare_data.py stores the ready-to-tokenize model input in the ``source``
    column (already the full input string). Accept, in priority order:
      * an existing ``text`` column;
      * a ``source`` column -> used directly as ``text``.

    Values are coerced to plain ``str`` (NaN/None -> ""), because the fast
    tokenizer rejects any non-string element with a cryptic
    ``TextEncodeInput must be Union[...]`` error.
    """

    def _as_text(series: pd.Series) -> pd.Series:
        return series.map(lambda v: "" if pd.isna(v) else str(v))

    if "text" in df.columns:
        df["text"] = _as_text(df["text"])
        return df
    cols = {c.lower(): c for c in df.columns}
    if "source" in cols:
        df["text"] = _as_text(df[cols["source"]])
        return df
    raise KeyError(
        f"{name} has no usable input column. Expected 'text' or 'source'. "
        f"Found: {list(df.columns)}"
    )


def prepare_from_excel(args):
    """Read the raw Excel from GCS, clean it, and return (train_df, val_df, maps)."""
    print(f"Reading spreadsheet: {args.excel_path}", flush=True)
    df = read_table(args.excel_path, sheet_name=args.sheet_name)
    print(f"  Loaded {len(df):,} raw rows.", flush=True)

    df = df.rename(
        columns={
            args.col_source: "source",
            args.col_diagnosis: "diagnosis",
            args.col_icd: "icd",
        }
    )
    df["source"] = df["source"].apply(clean_text)
    df["diagnosis"] = df["diagnosis"].apply(clean_text)
    df["icd"] = df["icd"].apply(clean_icd)

    before = len(df)
    df = df[(df["diagnosis"] != "") & (df["icd"] != "")]
    print(f"  Dropped {before - len(df):,} rows missing diagnosis or ICD.", flush=True)

    if args.drop_duplicates:
        df = df.drop_duplicates(subset=["source", "diagnosis", "icd"])
        print(f"  {len(df):,} rows after de-duplication.", flush=True)

    counts = df["icd"].value_counts()
    keep = counts[counts >= args.min_samples_per_class].index
    dropped_classes = len(counts) - len(keep)
    df = df[df["icd"].isin(keep)]
    print(
        f"  Removed {dropped_classes} ICD codes with < {args.min_samples_per_class} "
        f"samples. {len(df):,} rows / {df['icd'].nunique()} classes remain.",
        flush=True,
    )

    df["text"] = df.apply(lambda r: build_input_text(r["source"], r["diagnosis"]), axis=1)

    labels = sorted(df["icd"].unique())
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    df["label"] = df["icd"].map(label2id)
    maps = {"label2id": label2id, "id2label": id2label}

    val_fraction = min(0.5, args.val_size / len(df))
    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        random_state=args.seed,
        stratify=df["label"],
    )
    train_df = train_df[["text", "label", "icd"]].reset_index(drop=True)
    val_df = val_df[["text", "label", "icd"]].reset_index(drop=True)
    print(f"  Split -> train {len(train_df):,} / val {len(val_df):,}", flush=True)

    # Optionally persist the prepared split back to GCS for audit/reuse.
    if args.prepared_data_dir:
        base = args.prepared_data_dir.rstrip("/")
        write_csv(train_df, f"{base}/train.xlsx")
        write_csv(val_df, f"{base}/val.xlsx")
        write_json(maps, f"{base}/label_map.json")
        print(f"  Prepared data written to {base}/", flush=True)

    return train_df, val_df, maps


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    top5 = np.argsort(logits, axis=-1)[:, -5:]
    top5_acc = np.mean([lab in row for lab, row in zip(labels, top5)])
    return {
        "accuracy": accuracy_score(labels, preds),
        "top5_accuracy": top5_acc,
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def build_training_args(args, output_dir: str) -> TrainingArguments:
    common = dict(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        seed=args.seed,
        fp16=args.fp16,
    )
    try:
        return TrainingArguments(eval_strategy="epoch", **common)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **common)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()

    # --- Data input: EITHER a raw spreadsheet OR pre-split CSVs ---
    # Spreadsheet mode (preprocessing happens in the cloud):
    parser.add_argument("--excel_path", help="gs://icdpredictor123/train.xlsx")
    parser.add_argument("--sheet_name", default="Sheet1")
    parser.add_argument("--col_source", default="Source")
    parser.add_argument("--col_diagnosis", default="Diagnosis")
    parser.add_argument("--col_icd", default="ICD")
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--drop_duplicates", action="store_true")
    parser.add_argument("--min_samples_per_class", type=int, default=2)
    parser.add_argument(
        "--prepared_data_dir",
        default=None,
        help="Optional gs:// dir to save the generated train/val/label_map.",
    )
    # Pre-split CSV mode (preprocessing already done locally):
    parser.add_argument("--train_path")
    parser.add_argument("--val_path")
    parser.add_argument("--label_map_path")
    parser.add_argument(
        "--model_dir",
        default=os.environ.get("AIP_MODEL_DIR"),
        help="Output dir for the trained model. Defaults to Vertex AIP_MODEL_DIR.",
    )
    parser.add_argument(
        "--base_model",
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    )
    parser.add_argument("--num_epochs", type=float, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Mixed precision (GPU).")
    parser.add_argument(
        "--report_hpt",
        action="store_true",
        help="Report accuracy to the Vertex hyperparameter tuning service.",
    )
    args = parser.parse_args()

    if not args.model_dir:
        raise ValueError("No --model_dir and AIP_MODEL_DIR is unset.")

    # --- data + labels: spreadsheet mode OR pre-split CSV mode ---
    if args.excel_path:
        train_df, val_df, maps = prepare_from_excel(args)
    elif args.train_path and args.val_path and args.label_map_path:
        maps = read_json(args.label_map_path)
        train_df = ensure_text_column(read_table(args.train_path), "train_path")
        val_df = ensure_text_column(read_table(args.val_path), "val_path")
    else:
        raise ValueError(
            "Provide either --excel_path (spreadsheet mode) or all of "
            "--train_path/--val_path/--label_map_path (CSV mode)."
        )

    label2id = {k: int(v) for k, v in maps["label2id"].items()}
    id2label = {int(k): v for k, v in maps["id2label"].items()}
    num_labels = len(label2id)
    print(f"Loaded {num_labels} ICD-10 classes.", flush=True)
    print(f"Train rows: {len(train_df):,} | Val rows: {len(val_df):,}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False)
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Train under a local dir; upload at the end (transformers can't write gs://).
    with tempfile.TemporaryDirectory() as tmp:
        checkpoints_dir = os.path.join(tmp, "checkpoints")
        local_model_dir = os.path.join(tmp, "model")
        training_args = build_training_args(args, checkpoints_dir)

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

        print("\n=== Training ===", flush=True)
        trainer.train()

        print("\n=== Final validation metrics ===", flush=True)
        metrics = trainer.evaluate()
        for k, v in metrics.items():
            print(f"  {k}: {v}", flush=True)

        # Optional: report to Vertex hyperparameter tuning.
        if args.report_hpt:
            try:
                import hypertune

                hpt = hypertune.HyperTune()
                hpt.report_hyperparameter_tuning_metric(
                    hyperparameter_metric_tag="accuracy",
                    metric_value=float(metrics["eval_accuracy"]),
                )
            except Exception as exc:  # pragma: no cover
                print(f"hypertune reporting skipped: {exc}", flush=True)

        trainer.save_model(local_model_dir)
        tokenizer.save_pretrained(local_model_dir)
        # Persist the label map alongside the model for serving.
        with open(os.path.join(local_model_dir, "label_map.json"), "w") as fh:
            json.dump(maps, fh, indent=2)

        print(f"\nUploading model -> {args.model_dir}", flush=True)
        upload_dir(local_model_dir, args.model_dir)

    print("\n✅ Training complete.", flush=True)


if __name__ == "__main__":
    main()
