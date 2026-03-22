"""PyTorch + Gemma 2B LoRA fine-tuning for sequence classification.

This is the training container entry point. It:
1. Loads training config from GCS
2. Loads dataset from GCS
3. Fine-tunes Gemma 2B with LoRA adapters
4. Logs metrics to Vertex AI Experiments
5. Uploads model artifacts to GCS
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import yaml
from google.cloud import aiplatform, storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for PyTorch training."""
    parser = argparse.ArgumentParser(description="Train Gemma 2B LoRA model")
    parser.add_argument("--config", required=True, help="GCS path to training config YAML")
    parser.add_argument("--dataset", required=True, help="GCS path to dataset directory")
    parser.add_argument("--experiment", required=True, help="Vertex AI Experiment name")
    parser.add_argument("--output", default=None, help="GCS path for model artifacts")
    return parser.parse_args()


def download_from_gcs(gcs_path: str, local_path: str) -> str:
    """Download a GCS object to a local path."""
    client = storage.Client()
    # Parse gs://bucket/path
    parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(parts[0])
    blob = bucket.blob(parts[1])
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    return local_path


def load_config(gcs_path: str) -> dict:
    """Load training config from GCS."""
    local = download_from_gcs(gcs_path, "/tmp/training_config.yaml")
    with open(local) as f:
        return yaml.safe_load(f)


def load_dataset(gcs_path: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Load train/eval/test splits from GCS JSONL files."""
    splits = {}
    for split in ("train", "eval", "test"):
        local = f"/tmp/data/{split}.jsonl"
        download_from_gcs(f"{gcs_path}/{split}.jsonl", local)
        records = []
        with open(local) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        splits[split] = records
    return splits["train"], splits["eval"], splits["test"]


def train(config: dict, train_data: list[dict], eval_data: list[dict]) -> Path:
    """Fine-tune Gemma 2B with LoRA."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig as PeftLoraConfig
    from peft import TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    base_model = config.get("base_model", "google/gemma-2b")
    lora = config.get("lora", {})
    hyperparams = config.get("hyperparameters", {})

    logger.info("Loading base model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build label mapping from config
    label_schema = config.get("label_schema", {})
    all_labels = []
    for axis_labels in label_schema.values():
        all_labels.extend(axis_labels)
    label2id = {lbl: i for i, lbl in enumerate(sorted(set(all_labels)))}
    id2label = {i: lbl for lbl, i in label2id.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        torch_dtype=torch.float16,
    )

    # Apply LoRA
    peft_config = PeftLoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora.get("r", 16),
        lora_alpha=lora.get("alpha", 32),
        lora_dropout=lora.get("dropout", 0.1),
        target_modules=lora.get("target_modules", ["q_proj", "v_proj"]),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Prepare datasets
    def prepare(records):
        texts = [r.get("text", "") for r in records]
        # Use the first axis label for single-label classification
        first_axis = next(iter(label_schema)) if label_schema else None
        labels = []
        for r in records:
            lbl = r.get("labels", {})
            if isinstance(lbl, dict) and first_axis:
                labels.append(label2id.get(lbl.get(first_axis, ""), 0))
            else:
                labels.append(0)
        encoded = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
        encoded["labels"] = labels
        return Dataset.from_dict(encoded)

    train_ds = prepare(train_data)
    eval_ds = prepare(eval_data)

    output_dir = Path("/tmp/model_output")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=hyperparams.get("epochs", 3),
        per_device_train_batch_size=hyperparams.get("batch_size", 8),
        learning_rate=hyperparams.get("learning_rate", 2e-4),
        warmup_ratio=hyperparams.get("warmup_ratio", 0.1),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    return output_dir / "final"


def upload_artifacts(local_dir: Path, gcs_output: str) -> None:
    """Upload model artifacts to GCS."""
    client = storage.Client()
    parts = gcs_output.replace("gs://", "").split("/", 1)
    bucket = client.bucket(parts[0])
    prefix = parts[1].rstrip("/")

    for local_file in local_dir.rglob("*"):
        if local_file.is_file():
            blob_path = f"{prefix}/{local_file.relative_to(local_dir)}"
            bucket.blob(blob_path).upload_from_filename(str(local_file))
            logger.info("Uploaded %s", blob_path)


def main() -> None:
    """PyTorch training container entry point."""
    args = parse_args()

    project_id = os.environ.get("CLOUD_ML_PROJECT_ID", "i4g-ml")
    aiplatform.init(project=project_id, location="us-central1")

    logger.info("Loading config from %s", args.config)
    config = load_config(args.config)

    logger.info("Loading dataset from %s", args.dataset)
    train_data, eval_data, test_data = load_dataset(args.dataset)
    logger.info("Loaded %d train, %d eval, %d test records", len(train_data), len(eval_data), len(test_data))

    logger.info("Starting training")
    model_dir = train(config, train_data, eval_data)

    output = args.output or f"gs://i4g-ml-data/models/{args.experiment}/"
    logger.info("Uploading artifacts to %s", output)
    upload_artifacts(model_dir, output)

    logger.info("Training complete")


if __name__ == "__main__":
    main()
