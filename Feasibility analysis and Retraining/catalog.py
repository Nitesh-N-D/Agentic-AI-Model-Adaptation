"""
catalog.py
----------
Step 3: Update Model Catalog.
Enhancements: Migrated backend to SQLite DB for transactional safety, thread concurrency,
and robust error handling, preserving backwards compatibility.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class ModelVersionEntry:
    model_id: str
    version: int
    model_path: str
    created_at: str
    trigger_metrics: Dict[str, Any]
    validation_accuracy: float
    validation_f1: float
    status: str


class ModelCatalog:
    """An enterprise SQLite-backed Model Catalog."""

    def __init__(self, db_path: str = "./model_catalog.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS model_versions (
                        model_id TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        model_path TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        trigger_metrics TEXT NOT NULL,
                        validation_accuracy REAL NOT NULL,
                        validation_f1 REAL NOT NULL,
                        status TEXT NOT NULL,
                        PRIMARY KEY (model_id, version)
                    )
                """)
                conn.commit()
        except sqlite3.Error as e:
            logger.error("Database initialization failed: %s", e)
            raise

    def register_version(
        self,
        model_id: str,
        model_path: str,
        trigger_metrics: Dict[str, Any],
        validation_accuracy: float,
        validation_f1: float = 0.0,
        status: str = "pending_validation",
    ) -> ModelVersionEntry:
        """Registers a new auto-incremented model version."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT MAX(version) FROM model_versions WHERE model_id = ?", (model_id,)
                )
                row = cursor.fetchone()
                next_version = (row[0] + 1) if row and row[0] is not None else 1

                entry = ModelVersionEntry(
                    model_id=model_id,
                    version=next_version,
                    model_path=model_path,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    trigger_metrics=trigger_metrics,
                    validation_accuracy=validation_accuracy,
                    validation_f1=validation_f1,
                    status=status,
                )

                cursor.execute(
                    """
                    INSERT INTO model_versions 
                    (model_id, version, model_path, created_at, trigger_metrics, validation_accuracy, validation_f1, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.model_id,
                        entry.version,
                        entry.model_path,
                        entry.created_at,
                        json.dumps(entry.trigger_metrics),
                        entry.validation_accuracy,
                        entry.validation_f1,
                        entry.status,
                    ),
                )
                conn.commit()
                logger.info("Registered model %s v%d in SQLite catalog.", model_id, next_version)
                return entry
        except sqlite3.Error as e:
            logger.error("Failed to register model version: %s", e)
            raise

    def update_status(self, model_id: str, version: int, new_status: str) -> None:
        """Updates the deployment status of a registered version."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE model_versions SET status = ? WHERE model_id = ? AND version = ?",
                    (new_status, model_id, version),
                )
                conn.commit()
                if cursor.rowcount == 0:
                    raise ValueError(f"No entry found for {model_id} v{version}")
                logger.info("Updated %s v%d status to '%s'", model_id, version, new_status)
        except sqlite3.Error as e:
            logger.error("Failed to update status for %s v%d: %s", model_id, version, e)
            raise

    def get_latest(self, model_id: str) -> Optional[dict]:
        """Retrieves the latest registered version dictionary."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM model_versions WHERE model_id = ? ORDER BY version DESC LIMIT 1",
                (model_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            res["trigger_metrics"] = json.loads(res["trigger_metrics"])
            return res

    def list_versions(self, model_id: str) -> List[dict]:
        """Lists all registered versions for a model."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM model_versions WHERE model_id = ? ORDER BY version ASC", (model_id,)
            )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["trigger_metrics"] = json.loads(d["trigger_metrics"])
                results.append(d)
            return results