FROM python:3.11-slim

WORKDIR /app

# Reuse the serve container's dependencies (includes model loading code)
COPY containers/serve/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install batch-specific dependencies (BigQuery)
RUN pip install --no-cache-dir google-cloud-bigquery

# Copy the ml package source
COPY src/ /app/src/
ENV PYTHONPATH="/app/src"

# Copy the batch prediction entry point
COPY scripts/run_batch_prediction.py .

ENTRYPOINT ["python", "run_batch_prediction.py"]
