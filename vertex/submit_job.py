"""
Launch the PubMedBERT ICD-10 training job on Vertex AI Custom Training.

Prerequisites (run once, from this vertex/ folder):

  # 0. Authenticate + set your project
  gcloud auth login
  gcloud config set project YOUR_PROJECT_ID
  gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com

  # 1. Create a GCS bucket (skip if it exists) and upload the RAW spreadsheet
  gsutil mb -l us-central1 gs://YOUR_BUCKET
  gsutil cp "../../train.csv" \
      gs://YOUR_BUCKET/icd/raw/ICDDATA21_Venu_compare.xlsx

  # 2. Create an Artifact Registry repo (skip if it exists)
  gcloud artifacts repositories create icd-training \
      --repository-format=docker --location=us-central1

  # 3. Build + push the GPU training image (from this folder)
  gcloud builds submit --tag \
      us-central1-docker.pkg.dev/YOUR_PROJECT_ID/icd-training/pubmedbert:latest .

  # 4. Edit the CONFIG block below, then:
  python submit_job.py

The trained model (model.safetensors, tokenizer, label_map.json) is written to
gs://YOUR_BUCKET/icd/output/<job>/model/ via the AIP_MODEL_DIR env var.
"""
from google.cloud import aiplatform

# ============================ CONFIG ============================
PROJECT_ID = "qwiklabs-gcp-03-d9bd89368565"
REGION = "us-central1"
BUCKET = "questdxicdpredictor"  # no gs:// prefix

IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/icd-training/icdpredictorpubmedbert:latest"

GCS_PREFIX = f"gs://{BUCKET}"
BASE_OUTPUT_DIR = f"{GCS_PREFIX}/output"  # AIP_MODEL_DIR -> <this>/model/

# --- DATA MODE --------------------------------------------------------------
# "prepared":    you ran prepare_data.py and uploaded its 3 outputs
#                (train.xlsx / val.xlsx / label_map.json). No cleaning in cloud.
# "spreadsheet": upload the RAW workbook (Source/Diagnosis/ICD); cleaning +
#                train/val split happen inside the training job.
DATA_MODE = "prepared"

# prepared mode paths (upload prepare_data.py's outputs here):
TRAIN_PATH = f"{GCS_PREFIX}/data/train.xlsx"
VAL_PATH = f"{GCS_PREFIX}/data/val.xlsx"
LABEL_MAP_PATH = f"{GCS_PREFIX}/data/label_map.json"

# spreadsheet mode: RAW workbook + its schema
EXCEL_PATH = f"{GCS_PREFIX}/data/icdsourcedatapq.xlsx"
# Where the job saves the generated train/val/label_map (for audit/reuse).
PREPARED_DATA_DIR = f"{GCS_PREFIX}/prepared"
SHEET_NAME = "Sheet1"
COL_SOURCE = "Source"
COL_DIAGNOSIS = "Diagnosis"
COL_ICD = "ICD"
VAL_SIZE = "3000"
MIN_SAMPLES_PER_CLASS = "2"

# Compute.
# GPU needs quota (IAM & Admin > Quotas > "NVIDIA T4 GPUs"). If your project
# has no GPU quota you'll get "Accelerators are not supported for this project".
# Set USE_GPU = False to train on CPU (slower, but no quota needed).
USE_GPU = False  # flip to True once GPU quota is granted

if USE_GPU:
    MACHINE_TYPE = "n1-standard-8"
    ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
    ACCELERATOR_COUNT = 1
else:
    # CPU-only: use a high-memory machine so training doesn't run out of RAM.
    MACHINE_TYPE = "n1-highmem-8"
    ACCELERATOR_TYPE = None
    ACCELERATOR_COUNT = 0

# Training hyperparameters (mirror the local config defaults).
HPARAMS = {
    "num_epochs": "6",
    "batch_size": "16",
    "eval_batch_size": "32",
    "learning_rate": "2e-5",
    "weight_decay": "0.01",
    "warmup_ratio": "0.1",
    "max_length": "256",
}
# ================================================================


def build_args() -> list[str]:
    if DATA_MODE == "prepared":
        args = [
            f"--train_path={TRAIN_PATH}",
            f"--val_path={VAL_PATH}",
            f"--label_map_path={LABEL_MAP_PATH}",
        ]
    elif DATA_MODE == "spreadsheet":
        args = [
            f"--excel_path={EXCEL_PATH}",
            f"--sheet_name={SHEET_NAME}",
            f"--col_source={COL_SOURCE}",
            f"--col_diagnosis={COL_DIAGNOSIS}",
            f"--col_icd={COL_ICD}",
            f"--val_size={VAL_SIZE}",
            f"--min_samples_per_class={MIN_SAMPLES_PER_CLASS}",
            f"--prepared_data_dir={PREPARED_DATA_DIR}",
            "--drop_duplicates",
        ]
    else:
        raise ValueError(f"Unknown DATA_MODE: {DATA_MODE!r}")

    if USE_GPU:
        args.append("--fp16")  # mixed precision only works on GPU
    args += [f"--{k}={v}" for k, v in HPARAMS.items()]
    return args


def main() -> None:
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET}",
    )

    machine_spec = {"machine_type": MACHINE_TYPE}
    if ACCELERATOR_TYPE:
        machine_spec["accelerator_type"] = ACCELERATOR_TYPE
        machine_spec["accelerator_count"] = ACCELERATOR_COUNT

    worker_pool_specs = [
        {
            "machine_spec": machine_spec,
            "replica_count": 1,
            "container_spec": {
                "image_uri": IMAGE_URI,
                "args": build_args(),
            },
        }
    ]

    job = aiplatform.CustomJob(
        display_name="pubmedbert-icd10",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=BASE_OUTPUT_DIR,
    )

    print("Submitting Vertex AI CustomJob...")
    job.run(sync=True)
    print(f"\n✅ Done. Model written under: {BASE_OUTPUT_DIR}/model/")


if __name__ == "__main__":
    main()
