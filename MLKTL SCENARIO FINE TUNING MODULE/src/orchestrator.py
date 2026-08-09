"""
End-to-end reference run of the Week 1-4 design, on synthetic data.

Pipeline:
  Performance Monitoring (CSV inputs)
    -> Drift Analysis                      (Week 1)
    -> Dataset Selection                   (Week 2)
    -> Baseline Model Selection            (Week 2)
    -> Fine-Tuning Planning                (Week 3)
    -> Evaluation Gate                     (Week 4)
    -> MLKTLScenario handoff object        (Week 4 interface contract)

Run with:  python -m src.orchestrator
"""

import json
import os

import pandas as pd

from src.drift_analysis import population_stability_index, error_rate_shift, classify_drift
from src.dataset_selection import select_dataset
from src.baseline_selection import select_baseline_checkpoint, ModelCheckpoint
from src.finetuning_planner import build_plan
from src.evaluation_planner import run_evaluation_gate
from src.scenario_schema import MLKTLScenario, Strategy

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def strategy_from_drift(degradation_type, severity) -> Strategy:
    from src.scenario_schema import DegradationType, Severity

    if severity == Severity.HIGH and degradation_type == DegradationType.CONCEPT_DRIFT:
        return Strategy.FULL_RETRAIN
    if degradation_type == DegradationType.ANOMALY:
        return Strategy.ROLLBACK
    return Strategy.FINE_TUNE


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Step 3: Performance Monitoring (load logged feature snapshots) ----
    baseline_df = pd.read_csv(os.path.join(DATA_DIR, "sample_baseline_metrics.csv"))
    current_df = pd.read_csv(os.path.join(DATA_DIR, "sample_current_metrics.csv"))

    # ---- Step 6: Drift Analysis ----
    psi = population_stability_index(
        baseline_df["feature_value"].values, current_df["feature_value"].values
    )
    # Synthetic error-rate figures for this demo run.
    baseline_error_rate, current_error_rate = 0.05, 0.07
    err_delta = error_rate_shift(baseline_error_rate, current_error_rate)
    label_dist_shift = 0.05  # synthetic, low, for this demo

    degradation_type, severity = classify_drift(psi, err_delta, label_dist_shift)
    print(f"[Drift Analysis] PSI={psi:.4f}  error_rate_delta={err_delta:.2%}")
    print(f"[Drift Analysis] degradation_type={degradation_type.value}  severity={severity.value}")

    recommended_strategy = strategy_from_drift(degradation_type, severity)
    print(f"[Scenario Detection] recommended_strategy={recommended_strategy.value}")

    scenario = MLKTLScenario(
        scenario_id=MLKTLScenario.new_id(),
        detected_at=pd.Timestamp.now("UTC").isoformat(),
        degradation_type=degradation_type,
        severity=severity,
        affected_model_version="fraud-v3.2",
        recommended_strategy=recommended_strategy,
    )

    if recommended_strategy == Strategy.FINE_TUNE:
        # ---- Step 7: Dataset Identification ----
        scenario.dataset_spec = select_dataset(
            candidate_source="feature_store://transactions/2026-07-01_to_2026-08-01",
            drift_onset_window="2026-07-01 to 2026-08-01",
            label_quality_score=0.88,
            covers_current_distribution=True,
        )
        print(f"[Dataset Selection] {scenario.dataset_spec}")

        # ---- Step 8: Baseline Model Selection ----
        history = [
            ModelCheckpoint(version="fraud-v3.0", passed_validation_gate=True),
            ModelCheckpoint(version="fraud-v3.1", passed_validation_gate=True, parent_version="fraud-v3.0"),
            ModelCheckpoint(version="fraud-v3.2", passed_validation_gate=True, parent_version="fraud-v3.1"),
        ]
        scenario.baseline_checkpoint = select_baseline_checkpoint(history)
        print(f"[Baseline Selection] baseline_checkpoint={scenario.baseline_checkpoint}")

        # ---- Step 9-11: Fine-Tuning + Hyperparameter Planning ----
        scenario.fine_tuning_plan = build_plan(
            degradation_type, severity, compute_budget="sufficient"
        )
        print(f"[Fine-Tuning Plan] approach={scenario.fine_tuning_plan.approach.value}")

        # ---- Step 12-13: Evaluation + Validation Gate ----
        scenario.evaluation_report = run_evaluation_gate(
            baseline_metric=0.91,
            candidate_metric=0.935,           # synthetic post-fine-tune result
            baseline_regression_metric=0.90,
            candidate_regression_metric=0.905,
            shadow_kpi_delta_pct=1.8,
        )
        print(f"[Evaluation Gate] overall_pass={scenario.evaluation_report.overall_pass}")

    # ---- Step 14-18: Metadata + Results + Handoff ----
    scenario.lineage_metadata = {
        "generated_by": "mlktl_python_implementation.orchestrator",
        "design_doc_ref": "Weeks_1-4_Final_Deliverable.md",
    }

    out_path = os.path.join(OUTPUT_DIR, f"{scenario.scenario_id}.json")
    with open(out_path, "w") as f:
        json.dump(scenario.to_dict(), f, indent=2, default=str)

    print(f"\nScenario report written to: {out_path}")


if __name__ == "__main__":
    main()
