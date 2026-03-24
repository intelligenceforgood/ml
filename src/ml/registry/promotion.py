"""Model promotion between lifecycle stages.

Stage transitions:
  experimental → candidate (auto, eval gate passes)
  candidate → champion (manual approval)
  champion → retired (auto when replaced)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from google.cloud import aiplatform

from ml.config import get_settings
from ml.training.evaluation import EvalResult, NerEvalResult

logger = logging.getLogger(__name__)


@dataclass
class PromotionDecision:
    """Result of a promotion attempt."""

    approved: bool
    reason: str
    from_stage: str
    to_stage: str


def _passes_eval_gate(
    candidate: EvalResult,
    champion: EvalResult | None,
    *,
    max_regression: float = 0.05,
) -> tuple[bool, str]:
    """Check if a candidate model passes the eval gate vs the champion."""
    # First model always passes
    if champion is None:
        return True, "No existing champion — first model passes automatically"

    # Overall F1 must be >= champion
    if candidate.overall_f1 < champion.overall_f1:
        return False, (f"Candidate overall F1 ({candidate.overall_f1:.4f}) < " f"champion ({champion.overall_f1:.4f})")

    # No per-axis regression > max_regression
    for axis, cand_m in candidate.per_axis.items():
        if axis in champion.per_axis:
            champ_f1 = champion.per_axis[axis].f1
            regression = champ_f1 - cand_m.f1
            if regression > max_regression:
                return False, (f"Axis '{axis}' regressed by {regression:.4f} " f"(max allowed {max_regression:.4f})")

    return True, "Eval gate passed"


def _passes_ner_eval_gate(
    candidate: NerEvalResult,
    champion: NerEvalResult | None,
    *,
    max_regression: float = 0.05,
) -> tuple[bool, str]:
    """Check if a NER candidate passes the eval gate vs the champion.

    NER uses entity micro F1 as the primary metric and checks per-entity-type
    regression.
    """
    if champion is None:
        return True, "No existing NER champion — first model passes automatically"

    if candidate.micro_f1 < champion.micro_f1:
        return False, (f"Candidate entity micro F1 ({candidate.micro_f1:.4f}) < " f"champion ({champion.micro_f1:.4f})")

    # Per-entity-type regression check
    for entity_type, cand_m in candidate.per_entity_type.items():
        if entity_type in champion.per_entity_type:
            champ_f1 = champion.per_entity_type[entity_type].f1
            regression = champ_f1 - cand_m.f1
            if regression > max_regression:
                return False, (
                    f"Entity type '{entity_type}' regressed by {regression:.4f} " f"(max allowed {max_regression:.4f})"
                )

    return True, "NER eval gate passed"


def promote_model(
    model_name: str,
    candidate_metrics: EvalResult | NerEvalResult,
    champion_metrics: EvalResult | NerEvalResult | None = None,
    *,
    capability: str = "classification",
    target_stage: str = "candidate",
    max_regression: float = 0.05,
) -> PromotionDecision:
    """Attempt to promote a model to the next lifecycle stage."""
    settings = get_settings()
    aiplatform.init(
        project=settings.platform.project_id,
        location=settings.platform.region,
    )

    if target_stage == "candidate":
        if capability == "ner":
            passed, reason = _passes_ner_eval_gate(
                candidate_metrics,  # type: ignore[arg-type]
                champion_metrics,  # type: ignore[arg-type]
                max_regression=max_regression,
            )
        else:
            passed, reason = _passes_eval_gate(
                candidate_metrics,  # type: ignore[arg-type]
                champion_metrics,  # type: ignore[arg-type]
                max_regression=max_regression,
            )
        if not passed:
            logger.warning("Promotion rejected: %s", reason)
            return PromotionDecision(
                approved=False,
                reason=reason,
                from_stage="experimental",
                to_stage="candidate",
            )

        # Update model labels in Vertex AI Model Registry
        model = aiplatform.Model(model_name=model_name)
        model.update(labels={"stage": "candidate"})
        logger.info("Model %s promoted to candidate: %s", model_name, reason)
        return PromotionDecision(
            approved=True,
            reason=reason,
            from_stage="experimental",
            to_stage="candidate",
        )

    if target_stage == "champion":
        model = aiplatform.Model(model_name=model_name)
        model.update(labels={"stage": "champion"})
        logger.info("Model %s promoted to champion (manual approval)", model_name)
        return PromotionDecision(
            approved=True,
            reason="Manual promotion approved",
            from_stage="candidate",
            to_stage="champion",
        )

    raise ValueError(f"Unknown target stage: {target_stage}")
