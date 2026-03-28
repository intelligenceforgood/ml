FROM python:3.11-slim

WORKDIR /app

# Reuse the serve container's dependencies (includes model loading code)
COPY containers/serve/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install batch-specific dependencies (BigQuery)
RUN pip install --no-cache-dir google-cloud-bigquery

# Copy the ml package source and install CLI
COPY src/ /app/src/
COPY pyproject.toml /app/
ENV PYTHONPATH="/app/src"
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["i4g-ml", "serve", "batch"]
