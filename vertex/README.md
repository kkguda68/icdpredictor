# PubMedBERT ICD-10 → Vertex AI

GPU training package for the fine-tuned PubMedBERT ICD-10 classifier.

## What runs where

The training job can preprocess **and** train in the cloud. Two modes:

| Mode | Input | When |
|------|-------|------|
| Spreadsheet (default) | raw `.xlsx` in GCS | Upload the spreadsheet, let Vertex clean + split + train. |
| CSV | `train.csv`/`val.csv`/`label_map.json` in GCS | Preprocess locally with `prepare_data.py`, then train. |

Spreadsheet mode is the simplest: one upload, one job.

## Files
- `task.py` — training entrypoint. Reads a raw `.xlsx` (spreadsheet mode) or
  pre-split CSVs (CSV mode) from GCS; writes the model to `AIP_MODEL_DIR`.
- `Dockerfile` — CUDA 12.4 + PyTorch 2.6 GPU container.
- `requirements.txt` — training-only deps (incl. `openpyxl` for `.xlsx`).
- `submit_job.py` — launches the Vertex AI CustomJob (edit CONFIG block first).

## Quick start (spreadsheet mode)
```bash
# from this vertex/ folder — see submit_job.py header for full commands
gsutil cp "../../ICD Intelligence_RAG/ICDDATA21_Venu - compare.xlsx" \
  gs://YOUR_BUCKET/icd/raw/ICDDATA21_Venu_compare.xlsx

gcloud builds submit --tag \
  us-central1-docker.pkg.dev/YOUR_PROJECT_ID/icd-training/pubmedbert:latest .

# edit CONFIG in submit_job.py, then:
python submit_job.py
```

The job cleans the data, writes the split to `gs://YOUR_BUCKET/icd/prepared/`,
trains, and lands the model in `gs://YOUR_BUCKET/icd/output/model/`
(`model.safetensors`, tokenizer files, `label_map.json`).

## Serving (`serving/`)
A custom prediction container that loads the fine-tuned transformers model and
exposes the Vertex serving contract (`/health`, `/predict`).

- `serving/predictor.py` — FastAPI app; accepts `{"source", "diagnosis"}` or
  `{"text"}` instances and returns the top ICD code + top-5 with scores.
- `serving/Dockerfile` — GPU serving container (uvicorn + FastAPI).
- `serving/requirements.txt` — serving-only deps.
- `serving/deploy_model.py` — registers the model and deploys it to an endpoint.

```bash
# from vertex/serving/
gcloud builds submit --tag \
  us-central1-docker.pkg.dev/YOUR_PROJECT_ID/icd-training/pubmedbert-serve:latest .
# edit CONFIG in deploy_model.py, then:
python deploy_model.py
```

Call the endpoint with:
```json
{"instances": [{"source": "SKIN", "diagnosis": "basal cell carcinoma"}]}
```
