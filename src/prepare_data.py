"""
STEP 1 - Prepare the training data.

Reads the historical labelled Excel (Source / Diagnosis / ICD), cleans it,
builds the label <-> id maps, and writes stratified train / validation splits
to the ./data folder.

Run:
    python prepare_data.py
"""
import json
import os
import re
import fsspec

import pandas as pd
from sklearn.model_selection import train_test_split

import config


def clean_text(value) -> str:
    """Normalise a free-text cell: collapse whitespace, strip, lowercase-safe."""
    if pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)          # collapse runs of whitespace
    return text.strip().strip(":").strip()


def clean_icd(value) -> str:
    """Normalise an ICD code: uppercase, remove dots/spaces."""
    if pd.isna(value):
        return ""
    return str(value).upper().replace(".", "").replace(" ", "").strip()


def build_input_text(source: str, diagnosis: str) -> str:
    """Combine specimen source + diagnosis into a single model input string."""
    return f"Specimen Source: {source} [SEP] Final Diagnosis: {diagnosis}"


def main() -> None:
    if not config.DATA_DIR.startswith("gs://"):
        os.makedirs(config.DATA_DIR, exist_ok=True)

    print(f"1. Loading training data from:\n   {config.TRAINING_EXCEL}")
    # Note: For .xls files, ensure 'xlrd' is installed. For .xlsx, 'openpyxl'.
    try:
        df = pd.read_excel(config.TRAINING_EXCEL, sheet_name=config.TRAINING_SHEET)
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return
    print(f"   Loaded {len(df):,} raw rows.")

    # --- clean ---
    df = df.rename(
        columns={
            config.COL_SOURCE: "source",
            config.COL_DIAGNOSIS: "diagnosis",
            config.COL_ICD: "icd",
        }
    )
    df["source"] = df["source"].apply(clean_text)
    df["diagnosis"] = df["diagnosis"].apply(clean_text)
    df["icd"] = df["icd"].apply(clean_icd)

    # Drop rows with no source text, no diagnosis text or no label.
    before = len(df)
    df = df[(df["source"] != "") & (df["diagnosis"] != "") & (df["icd"] != "")]
    print(f"2. Dropped {before - len(df):,} rows missing source, diagnosis, or ICD.")

    # Optional: If you strictly want only the 'I' series codes (Circulatory system)
    # df = df[df["icd"].str.startswith("I")]
    # print(f"   Filtered for 'I' series codes. {len(df):,} rows remain.")

    # Optionally remove exact duplicates (same input + same label).
    if config.DROP_DUPLICATES:
        df = df.drop_duplicates(subset=["source", "diagnosis", "icd"])
        print(f"   {len(df):,} rows after de-duplication.")
    else:
        print(f"   {len(df):,} rows (duplicates kept).")

    # --- filter rare classes ---
    counts = df["icd"].value_counts()
    keep = counts[counts >= config.MIN_SAMPLES_PER_CLASS].index
    dropped_classes = len(counts) - len(keep)
    df = df[df["icd"].isin(keep)]
    print(
        f"3. Removed {dropped_classes} ICD codes with < "
        f"{config.MIN_SAMPLES_PER_CLASS} samples. "
        f"{len(df):,} rows / {df['icd'].nunique()} classes remain."
    )

    # --- build combined input text ---
    df["source"] = df.apply(
        lambda r: build_input_text(r["source"], r["diagnosis"]), axis=1
    )

    # --- label maps ---
    labels = sorted(df["icd"].unique())
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    df["label"] = df["icd"].map(label2id)

    with fsspec.open(config.LABEL_MAP_PATH, "w", encoding="utf-8") as fh:
        json.dump({"label2id": label2id, "id2label": id2label}, fh, indent=2)
    print(f"4. Saved label map ({len(labels)} classes) -> {config.LABEL_MAP_PATH}")

    # --- train / validation split ---
    # Stratify so every class is represented in both splits. Classes that are
    # too small to stratify the requested val size are handled by sklearn as
    # long as each class has >= 2 samples (guaranteed by the filter above).
    val_fraction = min(0.5, config.VAL_SIZE / len(df))
    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        random_state=config.RANDOM_SEED,
        stratify=df["label"],
    )

    train_df[["source", "label", "icd"]].to_excel(config.TRAIN_XLSX, index=False)
    val_df[["source", "label", "icd"]].to_excel(config.VAL_XLSX, index=False)

    print(
        f"5. Wrote splits:\n"
        f"   train -> {len(train_df):,} rows  ({config.TRAIN_XLSX})\n"
        f"   val   -> {len(val_df):,} rows  ({config.VAL_XLSX})"
    )
    print("\n✅ Data preparation complete.")


if __name__ == "__main__":
    main()
