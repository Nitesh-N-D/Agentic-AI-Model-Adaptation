"""
Week 4 deliverable: Evaluation protocol and validation/deployment gate,
per the design doc's evaluation table.
"""

from src.scenario_schema import EvaluationReport


def run_evaluation_gate(
    baseline_metric: float,
    candidate_metric: float,
    baseline_regression_metric: float,
    candidate_regression_metric: float,
    shadow_kpi_delta_pct: float = None,
    shadow_kpi_tolerance_pct: float = 5.0,
) -> EvaluationReport:
    """
    Three gates, all must pass for overall_pass=True:
      1. Offline evaluation: candidate must be >= baseline on the primary
         task metric (F1 / accuracy / AUC — caller supplies whichever
         applies).
      2. Regression check: candidate must not regress on segments the
         fine-tune wasn't targeting.
      3. Shadow gate (optional at call time): live shadow-traffic KPI must
         stay within tolerance of the target before canary/full rollout.
    """
    passed_offline = candidate_metric >= baseline_metric
    passed_regression = candidate_regression_metric >= baseline_regression_metric

    passed_shadow = None
    if shadow_kpi_delta_pct is not None:
        passed_shadow = abs(shadow_kpi_delta_pct) <= shadow_kpi_tolerance_pct

    overall = passed_offline and passed_regression and (passed_shadow in (None, True))

    return EvaluationReport(
        metrics={
            "baseline_metric": baseline_metric,
            "candidate_metric": candidate_metric,
            "baseline_regression_metric": baseline_regression_metric,
            "candidate_regression_metric": candidate_regression_metric,
            "shadow_kpi_delta_pct": shadow_kpi_delta_pct,
        },
        passed_offline_gate=passed_offline,
        passed_regression_check=passed_regression,
        passed_shadow_gate=passed_shadow,
        overall_pass=overall,
    )
