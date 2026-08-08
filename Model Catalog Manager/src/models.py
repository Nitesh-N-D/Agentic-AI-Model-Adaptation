"""
models.py
---------
Defines the data structure used to represent a model entry that flows
through the Model Catalog Manager (both incoming models and catalog
entries share this shape).
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import datetime


REQUIRED_FIELDS = ["model_id", "name", "version", "task_type", "architecture", "framework"]


class InvalidModelDataError(Exception):
    """Raised when incoming model metadata is missing required fields."""
    pass


@dataclass
class ModelMetadata:
    model_id: str
    name: str
    version: str
    task_type: str            # e.g. "text-classification", "summarization"
    architecture: str         # e.g. "transformer", "resnet", "lstm"
    framework: str            # e.g. "pytorch", "tensorflow"
    domain: str = "general"   # e.g. "finance", "healthcare", "general"
    size_params: str = "unknown"   # e.g. "7B", "125M"
    accuracy: Optional[float] = None
    source: str = "unknown"
    tags: list = field(default_factory=list)
    submitted_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    @staticmethod
    def from_dict(data: dict) -> "ModelMetadata":
        missing = [f for f in REQUIRED_FIELDS if f not in data or not data[f]]
        if missing:
            raise InvalidModelDataError(
                f"Incoming model is missing required field(s): {missing}"
            )
        known_keys = ModelMetadata.__dataclass_fields__.keys()
        clean = {k: v for k, v in data.items() if k in known_keys}
        return ModelMetadata(**clean)

    def to_dict(self) -> dict:
        return asdict(self)
