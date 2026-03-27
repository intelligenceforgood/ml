"""Evaluation, baseline, and framework comparison commands."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

logger = logging.getLogger(__name__)

eval_app = typer.Typer(help="Evaluation, baseline, and framework comparison.")


@eval_app.command("baseline")
def run_baseline(
    bucket: str = typer.Option("i4g-ml-data", "--bucket", "-b", help="GCS bucket for datasets."),
    train_path: str = typer.Option(
        "datasets/classification/v1/train.jsonl", "--train-path", help="GCS path to training JSONL."
    ),
    golden_path: str = typer.Option(
        "datasets/classification/golden/test.jsonl", "--golden-path", help="GCS path to golden test JSONL."
    ),
    output: Path = typer.Option(
        Path("data/baseline_result.json"), "--output", "-o", help="Local output path for baseline result."
    ),
) -> None:
    """Compute majority-class baseline benchmark against the golden test set.

    Always-predict-majority gives a minimum F1 floor that any trained model
    must exceed.
    """
    from collections import Counter

    from google.cloud import storage

    from ml.training.evaluation import compute_metrics

    def _load_jsonl(bucket_name: str, gcs_path: str) -> list[dict]:
        client = storage.Client()
        bkt = client.bucket(bucket_name)
        blob = bkt.blob(gcs_path)
        content = blob.download_as_text()
        return [json.loads(line) for line in content.strip().split("\n") if line.strip()]

    typer.echo(f"Loading training data from gs://{bucket}/{train_path}")
    train_records = _load_jsonl(bucket, train_path)
    typer.echo(f"Training records: {len(train_records)}")

    # Compute majority labels
    axis_counts: dict[str, Counter] = {}
    for rec in train_records:
        labels = rec.get("labels", {})
        for axis, code in labels.items():
            if axis not in axis_counts:
                axis_counts[axis] = Counter()
            axis_counts[axis][code] += 1

    majority_labels: dict[str, str] = {}
    for axis, counts in axis_counts.items():
        most_common = counts.most_common(1)[0]
        majority_labels[axis] = most_common[0]
        typer.echo(f"Majority {axis}: {most_common[0]} ({most_common[1]}/{sum(counts.values())})")

    typer.echo(f"Loading golden test set from gs://{bucket}/{golden_path}")
    golden_records = _load_jsonl(bucket, golden_path)
    typer.echo(f"Golden test records: {len(golden_records)}")

    predictions: list[dict[str, str]] = []
    ground_truth: list[dict[str, str]] = []
    for rec in golden_records:
        gt = rec.get("labels", {})
        if not gt:
            continue
        pred = {axis: majority_labels.get(axis, "") for axis in gt}
        predictions.append(pred)
        ground_truth.append(gt)

    result = compute_metrics(predictions, ground_truth)
    typer.echo(f"Majority-class baseline:\n{result.summary()}")

    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "baseline_type": "majority_class",
        "overall_f1": result.overall_f1,
        "overall_precision": result.overall_precision,
        "overall_recall": result.overall_recall,
        "total_samples": result.total_samples,
        "majority_labels": majority_labels,
        "per_axis": {
            axis: {"precision": m.precision, "recall": m.recall, "f1": m.f1, "support": m.support}
            for axis, m in result.per_axis.items()
        },
    }
    output.write_text(json.dumps(data, indent=2))
    typer.echo(f"Saved baseline result → {output}")


@eval_app.command("compare-frameworks")
def compare_frameworks(
    xgboost_metrics: str = typer.Option(
        ..., "--xgboost-metrics", help="Path or gs:// URI to XGBoost eval metrics JSON."
    ),
    pytorch_metrics: str = typer.Option(
        ..., "--pytorch-metrics", help="Path or gs:// URI to PyTorch eval metrics JSON."
    ),
) -> None:
    """Compare XGBoost vs PyTorch model metrics side by side.

    Prints formatted comparison table with per-axis F1 and recommendation.
    """

    def _load_metrics(path: str) -> dict:
        if path.startswith("gs://"):
            from google.cloud import storage

            without_scheme = path[5:]
            bucket_name, _, blob_path = without_scheme.partition("/")
            client = storage.Client()
            blob = client.bucket(bucket_name).blob(blob_path)
            return json.loads(blob.download_as_text())
        else:
            with open(path) as f:
                return json.load(f)

    xgb = _load_metrics(xgboost_metrics)
    pt = _load_metrics(pytorch_metrics)

    lines = [
        "=" * 70,
        "  Framework Comparison: XGBoost vs PyTorch (Gemma 2B LoRA)",
        "=" * 70,
        "",
        f"{'Metric':<30} {'XGBoost':>15} {'PyTorch':>15} {'Delta':>10}",
        "-" * 70,
    ]

    for key in ("overall_f1", "overall_precision", "overall_recall"):
        xgb_val = xgb.get(key, 0.0)
        pt_val = pt.get(key, 0.0)
        delta = pt_val - xgb_val
        sign = "+" if delta > 0 else ""
        lines.append(f"{key:<30} {xgb_val:>14.4f} {pt_val:>14.4f} {sign}{delta:>9.4f}")

    lines.append("")
    lines.append(f"{'Total samples':<30} {xgb.get('total_samples', 0):>15} {pt.get('total_samples', 0):>15}")
    lines.append("")

    xgb_axes = xgb.get("per_axis", {})
    pt_axes = pt.get("per_axis", {})
    all_axes = sorted(set(xgb_axes.keys()) | set(pt_axes.keys()))

    if all_axes:
        lines.append("Per-Axis F1:")
        lines.append(f"{'  Axis':<30} {'XGBoost':>15} {'PyTorch':>15} {'Delta':>10}")
        lines.append("-" * 70)
        for axis in all_axes:
            xgb_f1 = xgb_axes.get(axis, {}).get("f1", 0.0)
            pt_f1 = pt_axes.get(axis, {}).get("f1", 0.0)
            delta = pt_f1 - xgb_f1
            sign = "+" if delta > 0 else ""
            lines.append(f"  {axis:<28} {xgb_f1:>14.4f} {pt_f1:>14.4f} {sign}{delta:>9.4f}")

    lines.append("")
    lines.append("=" * 70)
    xgb_f1 = xgb.get("overall_f1", 0.0)
    pt_f1 = pt.get("overall_f1", 0.0)
    if pt_f1 > xgb_f1 + 0.02:
        lines.append("Recommendation: PyTorch (Gemma 2B LoRA) — higher accuracy.")
    elif xgb_f1 > pt_f1 + 0.02:
        lines.append("Recommendation: XGBoost — higher accuracy with lower cost.")
    else:
        lines.append("Recommendation: XGBoost (similar accuracy, faster + cheaper).")
    lines.append("=" * 70)

    typer.echo("\n".join(lines))
