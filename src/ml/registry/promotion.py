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
from ml.training.evaluation import EvalResult

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


def promote_model(
    model_name: str,
    candidate_metrics: EvalResult,
    champion_metrics: EvalResult | None = None,
    *,
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
        passed, reason = _passes_eval_gate(candidate_metrics, champion_metrics, max_regression=max_regression)
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
