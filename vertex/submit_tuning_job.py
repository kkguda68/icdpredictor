"""
Launch a Vertex AI HyperparameterTuningJob to find the best hyperparameters
for the PubMedBERT ICD-10 classifier.

This script defines a search space for parameters like learning rate and
number of epochs. Vertex AI will run multiple training trials and find the
combination that results in the best validation accuracy.
"""
from google.cloud import aiplatform
from google.cloud.aiplatform.hyperparameter_tuning import (
    HyperparameterTuningJob,
    ParameterSpec,
    DoubleParameterSpec,
    DiscreteParameterSpec,
)

# ============================ CONFIG ============================
PROJECT_ID = "qwiklabs-gcp-03-d9bd89368565"
REGION = "us-central1"
BUCKET = "questdxicdpredictor"  # no gs:// prefix

IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/icd-training/icdpredictorpubmedbert:latest"

GCS_PREFIX = f"gs://{BUCKET}"
BASE_OUTPUT_DIR = f"{GCS_PREFIX}/hpt_output"

# Data paths (assumes you are using pre-split data)
TRAIN_PATH = f"{GCS_PREFIX}/data/train.xlsx"
VAL_PATH = f"{GCS_PREFIX}/data/val.xlsx"
LABEL_MAP_PATH = f"{GCS_PREFIX}/data/label_map.json"

# Compute
MACHINE_TYPE = "n1-highmem-8"
ACCELERATOR_TYPE = None
ACCELERATOR_COUNT = 0
# ================================================================


def main() -> None:
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET}",
    )

    # --- Define the training container and its arguments ---
    # These are the fixed arguments for every trial.
    base_args = [
        f"--train_path={TRAIN_PATH}",
        f"--val_path={VAL_PATH}",
        f"--label_map_path={LABEL_MAP_PATH}",
        "--report_hpt",  # Crucial: tells task.py to report metrics
    ]

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
                "args": base_args,
            },
        }
    ]

    # --- Define the hyperparameter search space ---
    # Vertex will run trials with different values from these ranges.
    parameter_spec = {
        # The argument name must match what task.py expects (e.g., --learning_rate)
        "learning_rate": DoubleParameterSpec(min=1e-5, max=5e-5, scale="linear"),
        "num_epochs": DiscreteParameterSpec(values=[4, 6, 8], scale=None),
        "weight_decay": DoubleParameterSpec(min=0.01, max=0.1, scale="linear"),
    }

    # --- Define the metric to optimize ---
    # The metric_id must match the key reported by hypertune in task.py
    metric_spec = {"accuracy": "maximize"}

    # --- Create and run the HyperparameterTuningJob ---
    tuning_job = HyperparameterTuningJob(
        display_name="pubmedbert-icd10-hpt",
        worker_pool_specs=worker_pool_specs,
        parameter_spec=parameter_spec,
        metric_spec=metric_spec,
        max_trial_count=10,
        parallel_trial_count=2,
        base_output_dir=BASE_OUTPUT_DIR,
    )

    print("Submitting Vertex AI HyperparameterTuningJob...")
    tuning_job.run()

    # --- Print the results ---
    best_trial = sorted(
        tuning_job.trials, key=lambda t: t.final_measurement.metrics[0].value, reverse=True
    )[0]

    print("\n✅ Tuning complete.")
    print(f"Best trial: {best_trial.id}")
    print(f"  Accuracy: {best_trial.final_measurement.metrics[0].value:.4f}")
    print("  Optimal Hyperparameters:")
    for p in best_trial.parameters:
        print(f"    {p.parameter_id}: {p.value}")


if __name__ == "__main__":
    main()