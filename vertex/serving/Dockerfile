# Custom prediction container for serving PubMedBERT on a Vertex AI Endpoint.
# GPU base for low-latency inference; swap to a CPU torch image to save cost.
# torch >= 2.6 avoids known CVEs in older 2.2/2.4 wheels.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf \
    AIP_HTTP_PORT=8080 \
    AIP_HEALTH_ROUTE=/health \
    AIP_PREDICT_ROUTE=/predict

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predictor.py .

# Vertex sends traffic to AIP_HTTP_PORT; uvicorn serves the FastAPI app.
CMD ["sh", "-c", "uvicorn predictor:app --host 0.0.0.0 --port ${AIP_HTTP_PORT}"]
