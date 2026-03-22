FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    google-cloud-bigquery>=3.10 \
    sqlalchemy>=2.0 \
    pg8000>=1.30 \
    "cloud-sql-python-connector[pg8000]>=1.5" \
    pydantic>=2.5

# Copy the ml package source
COPY src/ /app/src/
COPY config/ /app/config/
ENV PYTHONPATH="/app/src"

ENTRYPOINT ["python", "-m", "ml.data.etl"]
