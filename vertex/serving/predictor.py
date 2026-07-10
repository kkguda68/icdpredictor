"""
Custom prediction server for the fine-tuned PubMedBERT ICD-10 classifier.

Implements the Vertex AI custom-container serving contract:
  * Health route  (GET)  -> AIP_HEALTH_ROUTE   (default /health)
  * Predict route (POST) -> AIP_PREDICT_ROUTE  (default /predict)
  * Listens on AIP_HTTP_PORT (default 8080)

The model artifacts are downloaded from AIP_STORAGE_URI (a gs:// path that
Vertex sets to the model directory) into a local folder at startup.

Request body (Vertex standard):
    {"instances": [
        {"source": "SKIN", "diagnosis": "basal cell carcinoma"},
        {"text": "Specimen Source: ... [SEP] Final Diagnosis: ..."}
    ]}

Response:
    {"predictions": [
        {"icd": "C449", "confidence": 0.97,
         "top5": [{"icd": "C449", "score": 0.97}, ...]}
    ]}
"""
import json
import os
import re

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")
STORAGE_URI = os.environ.get("AIP_STORAGE_URI", "")
MODEL_DIR = "/tmp/model"
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "256"))

app = FastAPI()

_model = None
_tokenizer = None
_id2label = {}
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _download_artifacts() -> str:
    """Return a local dir containing the model. Pull from GCS if needed."""
    if STORAGE_URI.startswith("gs://"):
        import fsspec

        fs = fsspec.filesystem("gcs")
        os.makedirs(MODEL_DIR, exist_ok=True)
        for remote in fs.find(STORAGE_URI):
            rel = remote[len(STORAGE_URI.replace("gs://", "").rstrip("/")) + 1 :]
            local_path = os.path.join(MODEL_DIR, rel)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            fs.get_file("gs://" + remote, local_path)
        return MODEL_DIR
    # Local path (or baked into the image) for testing.
    return STORAGE_URI or MODEL_DIR


def _clean_text(value) -> str:
    text = str(value or "")
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().strip(":").strip()


def _build_text(instance: dict) -> str:
    """Accept either a raw {"text": ...} or {"source", "diagnosis"} instance."""
    if "text" in instance and instance["text"]:
        return str(instance["text"])
    source = _clean_text(instance.get("source", ""))
    diagnosis = _clean_text(instance.get("diagnosis", ""))
    return f"Specimen Source: {source} [SEP] Final Diagnosis: {diagnosis}"


@app.on_event("startup")
def _load() -> None:
    global _model, _tokenizer, _id2label
    model_path = _download_artifacts()
    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    _model = AutoModelForSequenceClassification.from_pretrained(model_path)
    _model.to(_device)
    _model.eval()

    # Prefer the model's own id2label; fall back to the saved label_map.json.
    cfg_map = getattr(_model.config, "id2label", None)
    if cfg_map and all(str(v).startswith("LABEL_") is False for v in cfg_map.values()):
        _id2label = {int(k): v for k, v in cfg_map.items()}
    else:
        with open(os.path.join(model_path, "label_map.json")) as fh:
            maps = json.load(fh)
        _id2label = {int(k): v for k, v in maps["id2label"].items()}
    print(f"Loaded model on {_device} with {len(_id2label)} ICD classes.", flush=True)


@app.get(HEALTH_ROUTE)
def health():
    return {"status": "ok" if _model is not None else "loading"}


@app.post(PREDICT_ROUTE)
async def predict(request: Request):
    body = await request.json()
    instances = body.get("instances", [])
    if not isinstance(instances, list):
        return JSONResponse(status_code=400, content={"error": "instances must be a list"})

    texts = [_build_text(inst if isinstance(inst, dict) else {"text": inst}) for inst in instances]
    enc = _tokenizer(
        texts, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
    ).to(_device)

    with torch.no_grad():
        logits = _model(**enc).logits
        probs = torch.softmax(logits, dim=-1)

    k = min(5, probs.shape[-1])
    top_scores, top_idx = torch.topk(probs, k=k, dim=-1)

    predictions = []
    for scores, idx in zip(top_scores.tolist(), top_idx.tolist()):
        top5 = [{"icd": _id2label[i], "score": round(s, 4)} for s, i in zip(scores, idx)]
        predictions.append(
            {"icd": top5[0]["icd"], "confidence": top5[0]["score"], "top5": top5}
        )

    return {"predictions": predictions}
