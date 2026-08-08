"""
feasibility.py
--------------
Step 1: Check Retraining Feasibility.
Enhancements: Added logging, strict input boundary validation, and robust exception handling.
"""

from dataclasses import dataclass, field
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class FeasibilityReport:
    feasible: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "FEASIBLE" if self.feasible else "NOT FEASIBLE"
        lines = [f"Retraining feasibility: {status}"]
        lines += [f"  - {r}" for r in self.reasons]
        return "\n".join(lines)


def check_retraining_feasibility(
        new_data_size: int,
        drift_severity: float,  # Scale 0.0 to 1.0
        current_accuracy: float,  # Scale 0.0 to 1.0
        *,
        min_dataset_size: int = 500,
        max_drift_for_retrain: float = 0.70,
        min_accuracy_for_retrain: float = 0.40,
) -> FeasibilityReport:
    """Evaluates whether retraining is feasible based on dataset size, drift, and accuracy."""

    # Input boundary validation
    if not (0.0 <= drift_severity <= 1.0):
        raise ValueError(f"drift_severity must be in range [0.0, 1.0], got {drift_severity}")
    if not (0.0 <= current_accuracy <= 1.0):
        raise ValueError(f"current_accuracy must be in range [0.0, 1.0], got {current_accuracy}")
    if new_data_size < 0:
        raise ValueError(f"new_data_size cannot be negative, got {new_data_size}")

    reasons = []
    feasible = True

    # Data size check
    if new_data_size < min_dataset_size:
        feasible = False
        reasons.append(
            f"Insufficient new data: {new_data_size} samples (threshold >= {min_dataset_size})."
        )
    else:
        reasons.append(f"Dataset size OK: {new_data_size} samples (>= {min_dataset_size}).")

    # Drift severity check
    if drift_severity > max_drift_for_retrain:
        feasible = False
        reasons.append(
            f"Drift too severe: {drift_severity:.2f} (> {max_drift_for_retrain:.2f}). Requires full model regeneration."
        )
    else:
        reasons.append(f"Drift severity acceptable: {drift_severity:.2f} (<= {max_drift_for_retrain:.2f}).")

    # Current performance check
    if current_accuracy < min_accuracy_for_retrain:
        feasible = False
        reasons.append(
            f"Accuracy too low to recover via fine-tuning: {current_accuracy:.2f} (< {min_accuracy_for_retrain:.2f})."
        )
    else:
        reasons.append(f"Accuracy recoverable: {current_accuracy:.2f} (>= {min_accuracy_for_retrain:.2f}).")

    report = FeasibilityReport(
        feasible=feasible,
        reasons=reasons,
        metrics={
            "new_data_size": new_data_size,
            "drift_severity": drift_severity,
            "current_accuracy": current_accuracy,
        },
    )
    logger.info("Feasibility check completed. Result: %s", "FEASIBLE" if feasible else "NOT FEASIBLE")
    return report