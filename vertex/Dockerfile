# GPU training container for PubMedBERT ICD-10 fine-tuning on Vertex AI.
# Base image ships with CUDA 12.4 + cuDNN 9 + PyTorch 2.6
# (torch >= 2.6 avoids known CVEs in older 2.2/2.4 wheels).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY task.py .

# Vertex passes hyperparameters as CLI args after the entrypoint.
ENTRYPOINT ["python", "task.py"]
