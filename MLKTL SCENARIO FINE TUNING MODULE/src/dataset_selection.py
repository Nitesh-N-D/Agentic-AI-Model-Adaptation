"""
Week 2 deliverable: Dataset selection methodology.

Selects the data window that should be used to correct the model, per the
methodology: match the drift onset window, enforce a minimum label-quality
score, and require coverage of the current input distribution.
"""

from src.scenario_schema import DatasetSpec

MIN_LABEL_QUALITY_SCORE = 0.75


def select_dataset(
    candidate_source: str,
    drift_onset_window: str,
    label_quality_score: float,
    covers_current_distribution: bool,
) -> DatasetSpec:
    """
    Builds a DatasetSpec from candidate metadata. Raises if the candidate
    fails the minimum eligibility bar, so a bad dataset can never silently
    flow into the fine-tuning plan.
    """
    if label_quality_score < MIN_LABEL_QUALITY_SCORE:
        raise ValueError(
            f"Dataset '{candidate_source}' rejected: label_quality_score "
            f"{label_quality_score:.2f} is below minimum {MIN_LABEL_QUALITY_SCORE}"
        )
    if not covers_current_distribution:
        raise ValueError(
            f"Dataset '{candidate_source}' rejected: does not cover the "
            f"current input distribution (PSI check against live traffic failed)"
        )

    return DatasetSpec(
        source=candidate_source,
        time_window=drift_onset_window,
        label_quality_score=label_quality_score,
        covers_current_distribution=covers_current_distribution,
    )
