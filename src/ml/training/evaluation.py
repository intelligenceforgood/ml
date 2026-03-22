"""Model evaluation metrics and report generation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AxisMetrics:
    """Precision / Recall / F1 for a single taxonomy axis."""

    axis: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class EvalResult:
    """Full evaluation result across all axes."""

    overall_f1: float
    overall_precision: float
    overall_recall: float
    per_axis: dict[str, AxisMetrics] = field(default_factory=dict)
    total_samples: int = 0

    def summary(self) -> str:
        """Return a human-readable multi-line summary of metrics."""
        lines = [f"Overall F1={self.overall_f1:.4f}  P={self.overall_precision:.4f}  R={self.overall_recall:.4f}"]
        for axis, m in sorted(self.per_axis.items()):
            lines.append(f"  {axis}: F1={m.f1:.4f}  P={m.precision:.4f}  R={m.recall:.4f}  (n={m.support})")
        return "\n".join(lines)


def _safe_div(a: float, b: float) -> float:
    """Divide *a* by *b*, returning 0.0 when *b* is zero."""
    return a / b if b > 0 else 0.0


def compute_metrics(
    predictions: list[dict[str, str]],
    ground_truth: list[dict[str, str]],
) -> EvalResult:
    """Compute per-axis and overall P/R/F1.

    Each element is a dict mapping axis names to predicted/true label codes.
    Example: ``{"INTENT": "INTENT.ROMANCE", "CHANNEL": "CHANNEL.SOCIAL_MEDIA"}``
    """
    all_axes: set[str] = set()
    for gt in ground_truth:
        all_axes.update(gt.keys())

    axis_tp: dict[str, int] = {a: 0 for a in all_axes}
    axis_fp: dict[str, int] = {a: 0 for a in all_axes}
    axis_fn: dict[str, int] = {a: 0 for a in all_axes}
    axis_support: dict[str, int] = {a: 0 for a in all_axes}

    for pred, gt in zip(predictions, ground_truth, strict=False):
        for axis in all_axes:
            gt_label = gt.get(axis)
            pred_label = pred.get(axis)
            if gt_label is not None:
                axis_support[axis] += 1
                if pred_label == gt_label:
                    axis_tp[axis] += 1
                else:
                    axis_fn[axis] += 1
                    if pred_label is not None:
                        axis_fp[axis] += 1
            elif pred_label is not None:
                axis_fp[axis] += 1

    per_axis: dict[str, AxisMetrics] = {}
    f1_scores: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    weights: list[int] = []

    for axis in sorted(all_axes):
        tp, fp, fn = axis_tp[axis], axis_fp[axis], axis_fn[axis]
        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * p * r, p + r)
        support = axis_support[axis]
        per_axis[axis] = AxisMetrics(axis=axis, precision=p, recall=r, f1=f1, support=support)
        f1_scores.append(f1)
        precisions.append(p)
        recalls.append(r)
        weights.append(support)

    # Weighted average
    total_weight = sum(weights)
    if total_weight > 0:
        overall_f1 = sum(f * w for f, w in zip(f1_scores, weights, strict=False)) / total_weight
        overall_p = sum(p * w for p, w in zip(precisions, weights, strict=False)) / total_weight
        overall_r = sum(r * w for r, w in zip(recalls, weights, strict=False)) / total_weight
    else:
        overall_f1 = overall_p = overall_r = 0.0

    return EvalResult(
        overall_f1=overall_f1,
        overall_precision=overall_p,
        overall_recall=overall_r,
        per_axis=per_axis,
        total_samples=len(predictions),
    )
