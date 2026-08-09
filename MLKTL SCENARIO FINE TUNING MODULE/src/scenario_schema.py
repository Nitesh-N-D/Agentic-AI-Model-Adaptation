"""
Week 1 deliverable: MLKTL Scenario Schema.

Defines the structured, machine-readable objects this module produces and
hands off to the Agentic Decision Engine, Model Catalog, and Deployment
Manager. These mirror the schema documented in the Engineering Design
Document (Appendix B).
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class DegradationType(str, Enum):
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    LABEL_DRIFT = "label_drift"
    ANOMALY = "anomaly"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Strategy(str, Enum):
    FINE_TUNE = "fine_tune"
    INCREMENTAL_LEARNING = "incremental_learning"
    FULL_RETRAIN = "full_retrain"
    ROLLBACK = "rollback"


class FineTuningApproach(str, Enum):
    FULL = "full"
    LORA = "lora"
    ADAPTER = "adapter"
    PARTIAL_FREEZE = "partial_freeze"


@dataclass
class DatasetSpec:
    source: str
    time_window: str
    label_quality_score: float
    covers_current_distribution: bool


@dataclass
class HyperparameterRanges:
    learning_rate: tuple = (1e-5, 5e-4)
    epochs: tuple = (1, 5)
    batch_size: tuple = (8, 64)
    lora_rank: tuple = (4, 32)
    weight_decay: tuple = (0.0, 0.1)


@dataclass
class FineTuningPlan:
    approach: FineTuningApproach
    hyperparameter_ranges: HyperparameterRanges
    default_learning_rate: float
    default_epochs: int
    default_batch_size: int


@dataclass
class EvaluationReport:
    metrics: dict
    passed_offline_gate: bool
    passed_regression_check: bool
    passed_shadow_gate: Optional[bool]
    overall_pass: bool


@dataclass
class MLKTLScenario:
    scenario_id: str
    detected_at: str
    degradation_type: DegradationType
    severity: Severity
    affected_model_version: str
    recommended_strategy: Strategy
    dataset_spec: Optional[DatasetSpec] = None
    baseline_checkpoint: Optional[str] = None
    fine_tuning_plan: Optional[FineTuningPlan] = None
    evaluation_report: Optional[EvaluationReport] = None
    lineage_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def new_id(prefix: str = "scn") -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{prefix}_{stamp}"
