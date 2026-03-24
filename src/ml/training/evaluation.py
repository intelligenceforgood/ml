"""Model evaluation metrics and report generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


# ---------------------------------------------------------------------------
# NER Evaluation
# ---------------------------------------------------------------------------

# Entity types used in the I4G fraud domain
NER_ENTITY_TYPES = frozenset(["PERSON", "ORG", "CRYPTO_WALLET", "BANK_ACCOUNT", "PHONE", "EMAIL", "URL"])


@dataclass(frozen=True)
class EntityTypeMetrics:
    """Precision / Recall / F1 for a single entity type."""

    entity_type: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class NerEvalResult:
    """Full NER evaluation result with per-entity-type and aggregate metrics."""

    micro_f1: float
    micro_precision: float
    micro_recall: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    per_entity_type: dict[str, EntityTypeMetrics] = field(default_factory=dict)
    total_samples: int = 0

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        lines = [
            f"Micro  F1={self.micro_f1:.4f}  P={self.micro_precision:.4f}  R={self.micro_recall:.4f}",
            f"Macro  F1={self.macro_f1:.4f}  P={self.macro_precision:.4f}  R={self.macro_recall:.4f}",
        ]
        for etype, m in sorted(self.per_entity_type.items()):
            lines.append(f"  {etype}: F1={m.f1:.4f}  P={m.precision:.4f}  R={m.recall:.4f}  (n={m.support})")
        return "\n".join(lines)


@dataclass
class NerPrediction:
    """A single NER prediction with span-level entity annotations."""

    text: str
    entities: list[dict[str, Any]]  # [{"start": 0, "end": 10, "label": "PERSON"}]


@dataclass
class NerGolden:
    """A golden NER annotation."""

    text: str
    entities: list[dict[str, Any]]  # [{"start": 0, "end": 10, "label": "PERSON", "text": "John Doe"}]


def spans_to_bio_tags(text: str, spans: list[dict[str, Any]], tokens: list[str]) -> list[str]:
    """Convert character-offset entity spans to BIO tags aligned with token list.

    Args:
        text: The original text.
        spans: Entity spans with ``start``, ``end``, ``label`` keys (character offsets).
        tokens: Pre-tokenized token list (whitespace-split or subword).

    Returns:
        BIO tag list of the same length as *tokens*.
    """
    # Build a character-level label array
    char_labels = ["O"] * len(text)
    # Sort spans by start position to handle overlaps deterministically
    sorted_spans = sorted(spans, key=lambda s: s["start"])
    for span in sorted_spans:
        start, end, label = span["start"], span["end"], span["label"]
        for i in range(start, min(end, len(text))):
            char_labels[i] = label

    # Map tokens to character offsets via scanning
    tags: list[str] = []
    char_pos = 0
    prev_label: str | None = None

    for token in tokens:
        # Skip whitespace to find where this token starts
        while char_pos < len(text) and text[char_pos] == " ":
            char_pos += 1

        # Determine label for this token from its character span
        token_start = char_pos
        token_end = min(char_pos + len(token), len(text))

        # Majority vote: label that covers most of this token's chars
        label_counts: dict[str, int] = {}
        for i in range(token_start, token_end):
            lbl = char_labels[i]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        dominant_label = max(label_counts, key=label_counts.get) if label_counts else "O"

        if dominant_label == "O":
            tags.append("O")
            prev_label = None
        elif dominant_label != prev_label:
            tags.append(f"B-{dominant_label}")
            prev_label = dominant_label
        else:
            tags.append(f"I-{dominant_label}")

        char_pos = token_end

    return tags


def align_labels_with_tokens(
    labels: list[str],
    word_ids: list[int | None],
) -> list[str]:
    """Expand word-level BIO labels to subword-token-level labels.

    For subword tokenizers, multiple tokens may map to the same word.
    The first subword token gets the original label; continuation subword
    tokens get ``I-`` (if the word label is ``B-`` or ``I-``).

    Args:
        labels: Word-level BIO tags (length = number of words).
        word_ids: Per-subword-token word index (``None`` for special tokens).

    Returns:
        Subword-level BIO tags (length = number of subword tokens).
    """
    aligned: list[str] = []
    prev_word_id: int | None = None

    for word_id in word_ids:
        if word_id is None:
            # Special token ([CLS], [SEP], [PAD])
            aligned.append("O")
        elif word_id != prev_word_id:
            # First subword of a new word — keep original label
            aligned.append(labels[word_id] if word_id < len(labels) else "O")
        else:
            # Continuation subword — propagate entity as I-
            lbl = labels[word_id] if word_id < len(labels) else "O"
            if lbl.startswith("B-"):
                aligned.append(f"I-{lbl[2:]}")
            else:
                aligned.append(lbl)
        prev_word_id = word_id

    return aligned


def evaluate_ner(
    predictions: list[NerPrediction],
    golden: list[NerGolden],
) -> NerEvalResult:
    """Compute NER evaluation metrics using seqeval.

    Converts span annotations to BIO tags, then uses seqeval for
    per-entity-type and aggregate metrics.

    Args:
        predictions: Predicted entity spans per sample.
        golden: Ground-truth entity spans per sample.

    Returns:
        ``NerEvalResult`` with micro/macro averages and per-entity-type breakdown.
    """
    try:
        from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    except ImportError as err:
        raise ImportError("seqeval is required for NER evaluation: pip install seqeval>=1.2") from err

    all_true_tags: list[list[str]] = []
    all_pred_tags: list[list[str]] = []

    for pred, gt in zip(predictions, golden, strict=False):
        # Whitespace tokenization for evaluation (model-agnostic)
        tokens = pred.text.split()
        if not tokens:
            continue

        true_tags = spans_to_bio_tags(gt.text, gt.entities, tokens)
        pred_tags = spans_to_bio_tags(pred.text, pred.entities, tokens)

        all_true_tags.append(true_tags)
        all_pred_tags.append(pred_tags)

    if not all_true_tags:
        return NerEvalResult(
            micro_f1=0.0,
            micro_precision=0.0,
            micro_recall=0.0,
            macro_f1=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            total_samples=0,
        )

    # Aggregate metrics
    micro_f1 = f1_score(all_true_tags, all_pred_tags, average="micro", zero_division=0)
    micro_p = precision_score(all_true_tags, all_pred_tags, average="micro", zero_division=0)
    micro_r = recall_score(all_true_tags, all_pred_tags, average="micro", zero_division=0)
    macro_f1 = f1_score(all_true_tags, all_pred_tags, average="macro", zero_division=0)
    macro_p = precision_score(all_true_tags, all_pred_tags, average="macro", zero_division=0)
    macro_r = recall_score(all_true_tags, all_pred_tags, average="macro", zero_division=0)

    # Per-entity-type breakdown
    report_str = classification_report(all_true_tags, all_pred_tags, output_dict=True, zero_division=0)
    per_entity: dict[str, EntityTypeMetrics] = {}
    for entity_type, metrics in report_str.items():
        if isinstance(metrics, dict) and entity_type not in ("micro avg", "macro avg", "weighted avg"):
            per_entity[entity_type] = EntityTypeMetrics(
                entity_type=entity_type,
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1-score"],
                support=int(metrics["support"]),
            )

    return NerEvalResult(
        micro_f1=micro_f1,
        micro_precision=micro_p,
        micro_recall=micro_r,
        macro_f1=macro_f1,
        macro_precision=macro_p,
        macro_recall=macro_r,
        per_entity_type=per_entity,
        total_samples=len(predictions),
    )
