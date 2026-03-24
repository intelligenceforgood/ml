"""NER fine-tuning with token classification (BERT/DeBERTa).

This is the training container entry point for NER models. It:
1. Loads training config from GCS
2. Loads NER dataset from GCS (JSONL with character-offset entities)
3. Fine-tunes a token classification model
4. Logs per-entity-type F1 to Vertex AI Experiments
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

# BIO tags for I4G entity types
ENTITY_TYPES = ["PERSON", "ORG", "CRYPTO_WALLET", "BANK_ACCOUNT", "PHONE", "EMAIL", "URL"]
BIO_LABELS = ["O"] + [f"{prefix}-{et}" for et in ENTITY_TYPES for prefix in ("B", "I")]
LABEL2ID = {lbl: i for i, lbl in enumerate(BIO_LABELS)}
ID2LABEL = {i: lbl for lbl, i in LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NER token classification model")
    parser.add_argument("--config", required=True, help="GCS path to training config YAML")
    parser.add_argument("--dataset", required=True, help="GCS path to dataset directory")
    parser.add_argument("--experiment", required=True, help="Vertex AI Experiment name")
    parser.add_argument("--output", default=None, help="GCS path for model artifacts")
    return parser.parse_args()


def download_from_gcs(gcs_path: str, local_path: str) -> str:
    client = storage.Client()
    parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(parts[0])
    blob = bucket.blob(parts[1])
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    return local_path


def load_config(gcs_path: str) -> dict:
    local = download_from_gcs(gcs_path, "/tmp/training_config.yaml")
    with open(local) as f:
        return yaml.safe_load(f)


def load_ner_dataset(gcs_path: str, split: str) -> list[dict]:
    local = f"/tmp/data/{split}.jsonl"
    download_from_gcs(f"{gcs_path}/{split}.jsonl", local)
    records = []
    with open(local) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def spans_to_token_labels(
    text: str,
    entities: list[dict],
    tokenizer,
) -> tuple[list[int], list[int]]:
    """Convert character-offset entities to subword-token BIO label IDs.

    Returns (input_ids, label_ids) as aligned lists.
    """
    # Build character-level labels
    char_labels = ["O"] * len(text)
    for ent in sorted(entities, key=lambda e: e["start"]):
        start, end, label = ent["start"], ent["end"], ent["label"]
        for i in range(start, min(end, len(text))):
            char_labels[i] = label

    encoding = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=512)
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    label_ids = []
    prev_label = None

    for offset in offsets:
        start, end = offset
        if start == 0 and end == 0:
            # Special token
            label_ids.append(-100)
            prev_label = None
            continue

        # Majority label for this token span
        label_counts: dict[str, int] = {}
        for i in range(start, end):
            if i < len(char_labels):
                lbl = char_labels[i]
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

        dominant = max(label_counts, key=label_counts.get) if label_counts else "O"

        if dominant == "O":
            bio_tag = "O"
            prev_label = None
        elif dominant != prev_label:
            bio_tag = f"B-{dominant}"
            prev_label = dominant
        else:
            bio_tag = f"I-{dominant}"

        label_ids.append(LABEL2ID.get(bio_tag, 0))

    return input_ids, label_ids


def train(config: dict, train_data: list[dict], eval_data: list[dict]) -> Path:
    """Fine-tune a token classification model for NER."""
    import torch
    from datasets import Dataset
    from seqeval.metrics import f1_score
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    base_model = config.get("base_model", "dslim/bert-base-NER")
    hyperparams = config.get("hyperparameters", {})

    logger.info("Loading base model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Tokenize and align labels
    def _prepare_dataset(records: list[dict]) -> Dataset:
        all_input_ids = []
        all_attention_mask = []
        all_labels = []

        for rec in records:
            text = rec["text"]
            entities = rec.get("entities", [])
            input_ids, label_ids = spans_to_token_labels(text, entities, tokenizer)

            # Build attention mask
            attention_mask = [1] * len(input_ids)

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(label_ids)

        return Dataset.from_dict(
            {
                "input_ids": all_input_ids,
                "attention_mask": all_attention_mask,
                "labels": all_labels,
            }
        )

    train_ds = _prepare_dataset(train_data)
    eval_ds = _prepare_dataset(eval_data)

    use_gpu = torch.cuda.is_available()
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(BIO_LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )

    output_dir = Path("/tmp/ner_model_output")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=hyperparams.get("num_train_epochs", 5),
        per_device_train_batch_size=hyperparams.get("batch_size", 16),
        per_device_eval_batch_size=hyperparams.get("batch_size", 16),
        learning_rate=hyperparams.get("learning_rate", 5e-5),
        warmup_ratio=hyperparams.get("warmup_ratio", 0.1),
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=use_gpu,
        no_cuda=not use_gpu,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)

        true_labels = []
        pred_labels = []

        for pred_seq, label_seq in zip(predictions, labels, strict=False):
            true_tags = []
            pred_tags = []
            for p, lbl in zip(pred_seq, label_seq, strict=False):
                if lbl != -100:
                    true_tags.append(ID2LABEL.get(lbl, "O"))
                    pred_tags.append(ID2LABEL.get(p, "O"))
            true_labels.append(true_tags)
            pred_labels.append(pred_tags)

        micro_f1 = f1_score(true_labels, pred_labels, average="micro", zero_division=0)
        return {"f1": micro_f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )

    logger.info("Starting NER training: %d train, %d eval samples", len(train_ds), len(eval_ds))
    trainer.train()

    # Save model artifacts
    artifact_dir = Path("/tmp/ner_artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(artifact_dir)
    tokenizer.save_pretrained(artifact_dir)

    # Save label map
    with open(artifact_dir / "label_map.json", "w") as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, indent=2)

    # Save training config
    with open(artifact_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    return artifact_dir


def upload_artifacts(artifact_dir: Path, gcs_path: str) -> str:
    """Upload model artifacts to GCS."""
    client = storage.Client()
    parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(parts[0])
    prefix = parts[1] if len(parts) > 1 else ""

    for file_path in artifact_dir.rglob("*"):
        if file_path.is_file():
            blob_path = f"{prefix}/{file_path.relative_to(artifact_dir)}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(file_path))

    logger.info("Uploaded artifacts to %s", gcs_path)
    return gcs_path


def evaluate_and_log(
    config: dict,
    test_data: list[dict],
    artifact_dir: Path,
    experiment_name: str,
) -> dict:
    """Evaluate on test set and log metrics to Vertex AI."""
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
    model = AutoModelForTokenClassification.from_pretrained(artifact_dir)
    model.eval()

    import torch

    all_true = []
    all_pred = []

    for rec in test_data:
        text = rec["text"]
        entities = rec.get("entities", [])
        input_ids, label_ids = spans_to_token_labels(text, entities, tokenizer)

        with torch.no_grad():
            outputs = model(torch.tensor([input_ids]))
            predictions = outputs.logits.argmax(dim=-1)[0].tolist()

        true_tags = []
        pred_tags = []
        for p, lbl in zip(predictions, label_ids, strict=False):
            if lbl != -100:
                true_tags.append(ID2LABEL.get(lbl, "O"))
                pred_tags.append(ID2LABEL.get(p, "O"))

        all_true.append(true_tags)
        all_pred.append(pred_tags)

    micro_f1 = f1_score(all_true, all_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    micro_p = precision_score(all_true, all_pred, average="micro", zero_division=0)
    micro_r = recall_score(all_true, all_pred, average="micro", zero_division=0)

    metrics = {
        "entity_micro_f1": micro_f1,
        "entity_macro_f1": macro_f1,
        "entity_micro_precision": micro_p,
        "entity_micro_recall": micro_r,
        "test_samples": len(test_data),
    }

    # Log per entity type
    report = classification_report(all_true, all_pred, output_dict=True, zero_division=0)
    for etype, emetrics in report.items():
        if isinstance(emetrics, dict) and etype not in ("micro avg", "macro avg", "weighted avg"):
            metrics[f"{etype}_f1"] = emetrics["f1-score"]

    aiplatform.log_metrics(metrics)
    logger.info("Test metrics: %s", metrics)
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    project = config.get("project_id", "i4g-ml")
    region = config.get("region", "us-central1")
    aiplatform.init(project=project, location=region, experiment=args.experiment)
    aiplatform.start_run(args.experiment)

    train_data = load_ner_dataset(args.dataset, "train")
    eval_data = load_ner_dataset(args.dataset, "eval")
    test_data = load_ner_dataset(args.dataset, "test")
    logger.info("Loaded: train=%d, eval=%d, test=%d", len(train_data), len(eval_data), len(test_data))

    artifact_dir = train(config, train_data, eval_data)

    output_path = args.output or f"gs://i4g-ml-data/models/ner/{args.experiment}"
    upload_artifacts(artifact_dir, output_path)

    evaluate_and_log(config, test_data, artifact_dir, args.experiment)

    aiplatform.end_run()
    logger.info("NER training complete. Artifacts: %s", output_path)


if __name__ == "__main__":
    main()
