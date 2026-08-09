"""
Week 2 deliverable: Baseline model selection methodology.

Default: most recent stable (validation-gate-passed) checkpoint.
Exception: if the current degradation traces back to a specific prior
fine-tune, branch from the checkpoint immediately BEFORE that fine-tune
instead of compounding the error / catastrophic forgetting.
"""

from dataclasses import dataclass


@dataclass
class ModelCheckpoint:
    version: str
    passed_validation_gate: bool
    caused_current_degradation: bool = False
    parent_version: str = None


def select_baseline_checkpoint(catalog_history: list[ModelCheckpoint]) -> str:
    """
    catalog_history is ordered oldest -> newest, as returned by the Model
    Catalog for a given model lineage.
    """
    if not catalog_history:
        raise ValueError("No checkpoint history available from Model Catalog")

    latest = catalog_history[-1]

    if latest.caused_current_degradation:
        # Branch from the checkpoint immediately before the offending one.
        if latest.parent_version is None:
            raise ValueError(
                f"Checkpoint '{latest.version}' flagged as cause of degradation "
                f"but has no recorded parent to branch from"
            )
        parent = next((c for c in catalog_history if c.version == latest.parent_version), None)
        if parent is None or not parent.passed_validation_gate:
            raise ValueError(
                f"Cannot branch from parent '{latest.parent_version}': "
                f"not found or did not pass validation gate"
            )
        return parent.version

    if not latest.passed_validation_gate:
        # Walk backwards to the most recent stable checkpoint.
        for checkpoint in reversed(catalog_history[:-1]):
            if checkpoint.passed_validation_gate:
                return checkpoint.version
        raise ValueError("No stable (validation-gate-passed) checkpoint found in history")

    return latest.version
