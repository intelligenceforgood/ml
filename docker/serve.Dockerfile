FROM python:3.11-slim

WORKDIR /app

COPY containers/serve/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ml package source for serving
COPY src/ /app/src/
ENV PYTHONPATH="/app/src"

COPY containers/serve/serve.py .

EXPOSE 8080

ENTRYPOINT ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]
