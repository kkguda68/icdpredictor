"""
Register the trained PubMedBERT model and deploy it to a Vertex AI Endpoint
using the custom prediction container in vertex/serving/.

Prerequisites (run once, from the vertex/serving/ folder):

  # 1. Build + push the serving image
  gcloud builds submit --tag \
      us-central1-docker.pkg.dev/YOUR_PROJECT_ID/icd-training/pubmedbert-serve:latest .

  # 2. Confirm the trained model exists in GCS (written by the training job):
  gsutil ls gs://YOUR_BUCKET/icd/output/model/

  # 3. Edit the CONFIG block below, then:
  python deploy_model.py

After deployment, call the endpoint with:
    {"instances": [{"source": "SKIN", "diagnosis": "basal cell carcinoma"}]}
"""
import os
from google.cloud import aiplatform

# ============================ CONFIG ============================
PROJECT_ID = "qwiklabs-gcp-03-d9bd89368565"
REGION = "us-central1"
BUCKET = "questdxicdpredictor"  # no gs:// prefix
IMAGE_TAG = os.environ.get("IMAGE_TAG", "latest")

# Where the training job wrote the model (AIP_MODEL_DIR).
MODEL_ARTIFACT_URI = f"gs://{BUCKET}/output/model"

SERVE_IMAGE_URI = (
    f"{REGION}-docker.pkg.dev/{PROJECT_ID}/icd-training/pubmedbert-serve:{IMAGE_TAG}"
)

# Serving compute. Use a GPU for latency, or drop the accelerator for cost.
MACHINE_TYPE = "n1-standard-4"
ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
ACCELERATOR_COUNT = 1
MIN_REPLICAS = 1
MAX_REPLICAS = 1
# ================================================================


def main() -> None:
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET}")

    print("Uploading model to the Vertex Model Registry...")
    model = aiplatform.Model.upload(
        display_name="pubmedbert-icd10",
        artifact_uri=MODEL_ARTIFACT_URI,
        serving_container_image_uri=SERVE_IMAGE_URI,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_ports=[8080],
    )
    print(f"Model registered: {model.resource_name}")

    print("Deploying to an endpoint (this can take several minutes)...")
    deploy_kwargs = dict(
        machine_type=MACHINE_TYPE,
        min_replica_count=MIN_REPLICAS,
        max_replica_count=MAX_REPLICAS,
    )
    if ACCELERATOR_TYPE:
        deploy_kwargs["accelerator_type"] = ACCELERATOR_TYPE
        deploy_kwargs["accelerator_count"] = ACCELERATOR_COUNT

    endpoint = model.deploy(**deploy_kwargs)
    print(f"\n✅ Deployed. Endpoint: {endpoint.resource_name}")

    # Smoke test.
    result = endpoint.predict(
        instances=[{"source": "SKIN", "diagnosis": "basal cell carcinoma"}]
    )
    print("Sample prediction:", result.predictions)


if __name__ == "__main__":
    main()
