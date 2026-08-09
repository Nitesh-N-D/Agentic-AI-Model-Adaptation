"""
Week 3 deliverable: Fine-tuning strategy decision tree + hyperparameter
planning table, per the design doc's decision matrix.
"""

from src.scenario_schema import (
    DegradationType,
    Severity,
    FineTuningApproach,
    FineTuningPlan,
    HyperparameterRanges,
)


def choose_approach(
    degradation_type: DegradationType,
    severity: Severity,
    compute_budget: str,  # "sufficient" | "constrained"
) -> FineTuningApproach:
    """
    Decision tree:
      - Low severity data drift            -> LoRA / Adapter
      - Medium severity, sufficient budget -> Partial layer freeze
      - Medium severity, constrained budget-> LoRA / Adapter
      - High severity (concept drift)      -> escalate; caller should
        recommend FULL_RETRAIN strategy upstream, but if fine-tuning is
        still attempted as a stopgap, default to a full fine-tune.
    """
    if severity == Severity.LOW:
        return FineTuningApproach.LORA

    if severity == Severity.MEDIUM:
        return (
            FineTuningApproach.PARTIAL_FREEZE
            if compute_budget == "sufficient"
            else FineTuningApproach.LORA
        )

    # HIGH severity / concept drift: fine-tuning is a stopgap at best.
    return FineTuningApproach.FULL


def build_plan(
    degradation_type: DegradationType,
    severity: Severity,
    compute_budget: str,
) -> FineTuningPlan:
    approach = choose_approach(degradation_type, severity, compute_budget)
    ranges = HyperparameterRanges()

    return FineTuningPlan(
        approach=approach,
        hyperparameter_ranges=ranges,
        default_learning_rate=2e-5,
        default_epochs=3,
        default_batch_size=16,
    )
