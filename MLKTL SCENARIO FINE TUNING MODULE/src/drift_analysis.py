"""
Week 1 deliverable: Degradation detection & drift analysis.

Implements the metrics named in the design doc (PSI, KL-divergence) and a
classification function that maps drift magnitude to a DegradationType +
Severity, per the Week 1 taxonomy and threshold table.
"""

import numpy as np
from src.scenario_schema import DegradationType, Severity

PSI_MODERATE_THRESHOLD = 0.2
PSI_SEVERE_THRESHOLD = 0.25


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    PSI compares a baseline ("expected") distribution to a current
    ("actual") one. PSI > 0.2 is commonly treated as a moderate shift,
    PSI > 0.25 as severe (industry rule-of-thumb, e.g. used in credit-risk
    monitoring).
    """
    breakpoints = np.linspace(0, 100, bins + 1)
    edges = np.percentile(expected, breakpoints)
    edges[0], edges[-1] = -np.inf, np.inf

    expected_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=edges)[0] / len(actual)

    # avoid div-by-zero / log(0)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def kl_divergence(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """KL-divergence between binned baseline and current distributions."""
    edges = np.histogram_bin_edges(np.concatenate([expected, actual]), bins=bins)
    p = np.histogram(expected, bins=edges)[0] / len(expected)
    q = np.histogram(actual, bins=edges)[0] / len(actual)
    p = np.clip(p, 1e-6, None)
    q = np.clip(q, 1e-6, None)
    return float(np.sum(p * np.log(p / q)))


def error_rate_shift(baseline_error_rate: float, current_error_rate: float) -> float:
    """Simple proxy for concept drift: change in model error rate."""
    if baseline_error_rate == 0:
        return float("inf") if current_error_rate > 0 else 0.0
    return (current_error_rate - baseline_error_rate) / baseline_error_rate


def classify_drift(
    psi: float,
    error_rate_delta: float,
    label_distribution_shift: float,
) -> tuple[DegradationType, Severity]:
    """
    Maps raw drift signals to (DegradationType, Severity) per the Week 1
    taxonomy:
      - Large error-rate shift with modest input PSI -> concept drift
        (the input->label relationship changed, not just the inputs).
      - Large PSI with stable error rate -> data drift.
      - Large label_distribution_shift alone -> label drift.
      - Small, short-lived signals -> anomaly (caller should also check
        persistence over the rolling window; this function only judges
        magnitude).
    """
    if error_rate_delta > 0.30 and psi < PSI_MODERATE_THRESHOLD:
        degradation_type = DegradationType.CONCEPT_DRIFT
    elif psi >= PSI_SEVERE_THRESHOLD:
        degradation_type = DegradationType.DATA_DRIFT
    elif label_distribution_shift >= PSI_MODERATE_THRESHOLD:
        degradation_type = DegradationType.LABEL_DRIFT
    else:
        degradation_type = DegradationType.ANOMALY

    severity_score = max(psi, abs(error_rate_delta), label_distribution_shift)
    if severity_score >= PSI_SEVERE_THRESHOLD or error_rate_delta > 0.30:
        severity = Severity.HIGH
    elif severity_score >= PSI_MODERATE_THRESHOLD:
        severity = Severity.MEDIUM
    else:
        severity = Severity.LOW

    return degradation_type, severity
