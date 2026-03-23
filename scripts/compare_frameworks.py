"""Compare XGBoost vs PyTorch results on the same dataset.

Usage:
    conda run -n ml python scripts/compare_frameworks.py \\
        --xgboost-metrics gs://i4g-ml-data/models/xgboost-v1/eval_metrics.json \\
        --pytorch-metrics gs://i4g-ml-data/models/gemma2b-v1/eval_metrics.json

Or with local JSON files:
    python scripts/compare_frameworks.py \\
        --xgboost-metrics /tmp/xgb_metrics.json \\
        --pytorch-metrics /tmp/pt_metrics.json
"""

from __future__ import annotations

import argparse
import json


def _load_metrics(path: str) -> dict:
    """Load metrics from local file or GCS URI."""
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


def compare(xgb_metrics: dict, pt_metrics: dict) -> str:
    """Compare XGBoost vs PyTorch metrics and return a formatted report."""
    lines = [
        "=" * 70,
        "  Framework Comparison: XGBoost vs PyTorch (Gemma 2B LoRA)",
        "=" * 70,
        "",
        f"{'Metric':<30} {'XGBoost':>15} {'PyTorch':>15} {'Delta':>10}",
        "-" * 70,
    ]

    # Overall metrics
    for key in ("overall_f1", "overall_precision", "overall_recall"):
        xgb_val = xgb_metrics.get(key, 0.0)
        pt_val = pt_metrics.get(key, 0.0)
        delta = pt_val - xgb_val
        sign = "+" if delta > 0 else ""
        lines.append(f"{key:<30} {xgb_val:>14.4f} {pt_val:>14.4f} {sign}{delta:>9.4f}")

    lines.append("")
    lines.append(
        f"{'Total samples':<30} {xgb_metrics.get('total_samples', 0):>15} {pt_metrics.get('total_samples', 0):>15}"
    )
    lines.append("")

    # Per-axis breakdown
    xgb_axes = xgb_metrics.get("per_axis", {})
    pt_axes = pt_metrics.get("per_axis", {})
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

    # Recommendation
    lines.append("")
    lines.append("=" * 70)
    xgb_f1 = xgb_metrics.get("overall_f1", 0.0)
    pt_f1 = pt_metrics.get("overall_f1", 0.0)

    if pt_f1 > xgb_f1 + 0.02:
        lines.append("Recommendation: PyTorch (Gemma 2B LoRA) — higher accuracy.")
    elif xgb_f1 > pt_f1 + 0.02:
        lines.append("Recommendation: XGBoost — higher accuracy with lower cost.")
    else:
        lines.append("Recommendation: XGBoost (similar accuracy, faster + cheaper).")

    lines.append("")
    lines.append("Framework Selection Criteria:")
    lines.append("  - XGBoost: fast training (<10 min), CPU-only, low cost, good for tabular features")
    lines.append("  - PyTorch: leverages raw text, higher accuracy ceiling, requires GPU ($)")
    lines.append("  - Use XGBoost for rapid iteration; PyTorch for production accuracy")
    lines.append("=" * 70)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare XGBoost vs PyTorch metrics")
    parser.add_argument("--xgboost-metrics", required=True, help="Path or GCS URI to XGBoost eval metrics JSON")
    parser.add_argument("--pytorch-metrics", required=True, help="Path or GCS URI to PyTorch eval metrics JSON")
    args = parser.parse_args()

    xgb = _load_metrics(args.xgboost_metrics)
    pt = _load_metrics(args.pytorch_metrics)
    print(compare(xgb, pt))


if __name__ == "__main__":
    main()
