"""
Central configuration for the PubMedBERT ICD-10 fine-tuning pipeline.

Edit the paths / hyper-parameters here; every other script imports from this file
so there is a single source of truth.
"""
import os

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR = "gs://questicdpredictor"

# Use f-strings instead of os.path.join to ensure correct forward slashes for GCS
DATA_DIR = f"{BASE_DIR}/data"
TRAINING_EXCEL = f"{DATA_DIR}/icdsourcedatapq.xlsx"
TRAINING_SHEET = "Sheet1"

# Column names inside the training Excel (note the original typo "Diagnosis").
COL_SOURCE = "Source"
COL_DIAGNOSIS = "Diagnosis"
COL_ICD = "ICD"

# Where prepared splits + the trained model + label maps are written.

MODEL_DIR = f"{BASE_DIR}/model"
LABEL_MAP_PATH = f"{DATA_DIR}/label_map.json"
TRAIN_XLSX = f"{DATA_DIR}/train.xlsx"
VAL_XLSX = f"{DATA_DIR}/val.xlsx"

# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
# PubMedBERT fine-tuned on biomedical text. Good default for medical NLP.
BASE_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# ---------------------------------------------------------------------------
# DATA SPLIT
# ---------------------------------------------------------------------------
# User wanted ~10,000 train / ~3,000 validation. We express this as a ratio so
# it still works if the row count changes.
VAL_SIZE = 500          # number of rows held out for validation
RANDOM_SEED = 42

# The raw Excel contains many exact-duplicate rows. Set True to keep only unique
# (source, diagnosis, icd) rows (cleaner, ~8.9k rows). Set False to keep all
# rows, which yields ~10k+ training examples as originally requested.
# NOTE: True is strongly recommended - it prevents the same row appearing in
# both train and validation (data leakage), so reported accuracy is honest.
DROP_DUPLICATES = True

# Drop ICD codes that appear fewer than this many times in the whole dataset.
# Classes with a single example can't be learned or evaluated meaningfully.
MIN_SAMPLES_PER_CLASS = 2

# ---------------------------------------------------------------------------
# TRAINING HYPER-PARAMETERS
# ---------------------------------------------------------------------------
MAX_LENGTH = 256         # token cap for the [Source] + [Diagnosis] text
NUM_EPOCHS = 12
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
