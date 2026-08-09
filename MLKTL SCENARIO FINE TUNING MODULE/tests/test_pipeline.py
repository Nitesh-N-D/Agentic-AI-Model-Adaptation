import numpy as np
import pytest

from src.drift_analysis import population_stability_index, classify_drift
from src.dataset_selection import select_dataset
from src.baseline_selection import select_baseline_checkpoint, ModelCheckpoint
from src.finetuning_planner import build_plan, choose_approach
from src.evaluation_planner import run_evaluation_gate
from src.scenario_schema import DegradationType, Severity, FineTuningApproach


def test_psi_zero_for_identical_distributions():
    data = np.random.normal(0, 1, 1000)
    psi = population_stability_index(data, data.copy())
    assert psi < 1e-6


def test_psi_positive_for_shifted_distribution():
    baseline = np.random.normal(0, 1, 2000)
    shifted = np.random.normal(3, 1, 2000)
    psi = population_stability_index(baseline, shifted)
    assert psi > 0.25


def test_classify_drift_high_severity_concept_drift():
    dtype, severity = classify_drift(psi=0.05, error_rate_delta=0.5, label_distribution_shift=0.02)
    assert dtype == DegradationType.CONCEPT_DRIFT
    assert severity == Severity.HIGH


def test_dataset_selection_rejects_low_quality():
    with pytest.raises(ValueError):
        select_dataset("src", "window", label_quality_score=0.4, covers_current_distribution=True)


def test_dataset_selection_accepts_valid_candidate():
    spec = select_dataset("src", "window", label_quality_score=0.9, covers_current_distribution=True)
    assert spec.label_quality_score == 0.9


def test_baseline_selection_defaults_to_latest_stable():
    history = [
        ModelCheckpoint(version="v1", passed_validation_gate=True),
        ModelCheckpoint(version="v2", passed_validation_gate=True, parent_version="v1"),
    ]
    assert select_baseline_checkpoint(history) == "v2"


def test_baseline_selection_branches_off_offending_checkpoint():
    history = [
        ModelCheckpoint(version="v1", passed_validation_gate=True),
        ModelCheckpoint(
            version="v2", passed_validation_gate=True, parent_version="v1",
            caused_current_degradation=True,
        ),
    ]
    assert select_baseline_checkpoint(history) == "v1"


def test_finetuning_approach_low_severity_uses_lora():
    approach = choose_approach(DegradationType.DATA_DRIFT, Severity.LOW, "sufficient")
    assert approach == FineTuningApproach.LORA


def test_evaluation_gate_pass():
    report = run_evaluation_gate(
        baseline_metric=0.9, candidate_metric=0.92,
        baseline_regression_metric=0.88, candidate_regression_metric=0.89,
        shadow_kpi_delta_pct=2.0,
    )
    assert report.overall_pass is True


def test_evaluation_gate_fail_on_regression():
    report = run_evaluation_gate(
        baseline_metric=0.9, candidate_metric=0.92,
        baseline_regression_metric=0.88, candidate_regression_metric=0.80,
    )
    assert report.overall_pass is False
