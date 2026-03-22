FROM python:3.11-slim

WORKDIR /app

COPY containers/train-xgboost/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY containers/train-xgboost/train.py .

ENTRYPOINT ["python", "train.py"]
