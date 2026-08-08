"""
handoff.py
----------
Step 4: Handoff Decision Gate.
Enhancements: Evaluates multiple metrics (Accuracy AND F1-score), dynamic margins,
and structured decision logging.
"""

from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class HandoffDecision:
    action: str  # "deploy_to_team3" | "operator_review"
    reason: str
    model_path: str


def decide_handoff(
        *,
        feasible: bool,
        model_path: str,
        new_metrics: dict,
        previous_accuracy: float,
        previous_f1: float = 0.0,
        min_improvement_margin: float = 0.02,
) -> HandoffDecision:
    """Determines whether to auto-deploy to Team 3 or escalate for human review."""

    if not feasible:
        reason = "Retraining was not feasible. Escalating directly to operator."
        logger.info("Handoff decision: operator_review | Reason: %s", reason)
        return HandoffDecision(action="operator_review", reason=reason, model_path=model_path)

    new_acc = new_metrics.get("accuracy", 0.0)
    new_f1 = new_metrics.get("f1", 0.0)

    acc_delta = new_acc - previous_accuracy
    f1_delta = new_f1 - previous_f1

    # Check for regression
    if acc_delta < 0 or (previous_f1 > 0 and f1_delta < 0):
        reason = f"Model regressed vs baseline. Acc Delta: {acc_delta:.3f}, F1 Delta: {f1_delta:.3f}."
        logger.warning("Handoff decision: operator_review | Reason: %s", reason)
        return HandoffDecision(action="operator_review", reason=reason, model_path=model_path)

    # Check for confidence margin improvement across primary metrics
    if acc_delta >= min_improvement_margin:
        reason = f"Performance improved over baseline (Acc Gain: +{acc_delta:.3f} >= {min_improvement_margin}). Auto-approved."
        logger.info("Handoff decision: deploy_to_team3 | Reason: %s", reason)
        return HandoffDecision(action="deploy_to_team3", reason=reason, model_path=model_path)

    reason = f"Gain (+{acc_delta:.3f}) is within noisy margin threshold ({min_improvement_margin}). Routing to operator."
    logger.info("Handoff decision: operator_review | Reason: %s", reason)
    return HandoffDecision(action="operator_review", reason=reason, model_path=model_path)