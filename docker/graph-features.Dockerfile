FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "apache-beam[gcp]>=2.55,<3" \
    "networkx>=3.2,<4" \
    google-cloud-bigquery>=3.10 \
    pydantic>=2.5

# Copy the ml package source
COPY src/ /app/src/
COPY config/ /app/config/
ENV PYTHONPATH="/app/src"

ENTRYPOINT ["python", "-m", "ml.data.graph_features"]
