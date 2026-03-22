FROM python:3.11-slim AS base

WORKDIR /app

# Install PyTorch CPU first (GPU image will be used on Vertex AI via base image override)
RUN pip install --no-cache-dir torch>=2.1 --index-url https://download.pytorch.org/whl/cpu

COPY containers/train-pytorch/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY containers/train-pytorch/train.py .

ENTRYPOINT ["python", "train.py"]
