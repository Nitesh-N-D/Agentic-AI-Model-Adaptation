"""
version_manager.py
-------------------
Responsibility 4 (part A): Maintain model versions.

Keeps a separate version_history.json file so that whenever an existing
catalog entry is replaced/updated, the previous version is archived
rather than silently lost.
"""

import json
import os
import datetime


class VersionManagerError(Exception):
    pass


class VersionManager:
    def __init__(self, history_path: str):
        self.history_path = history_path
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        if not os.path.exists(self.history_path):
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump({"history": []}, f, indent=2)

    def _load(self) -> dict:
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                return json.loads(content) if content else {"history": []}
        except json.JSONDecodeError as e:
            raise VersionManagerError(f"Version history file is corrupted: {e}")

    def _save(self, data: dict):
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def archive_previous_version(self, old_model: dict, replaced_by: dict, match_type: str):
        """Store a snapshot of the model entry that is about to be replaced."""
        data = self._load()
        data["history"].append({
            "archived_model": old_model,
            "replaced_by_model_id": replaced_by["model_id"],
            "replaced_by_version": replaced_by["version"],
            "match_type": match_type,
            "archived_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        self._save(data)

    def get_history_for(self, model_id: str) -> list:
        data = self._load()
        return [h for h in data["history"] if h["archived_model"]["model_id"] == model_id]

    @staticmethod
    def next_version(old_version: str) -> str:
        """
        Best-effort semantic-ish version bump.
        '1.0' -> '1.1', '2.3.1' -> '2.3.2', non-numeric -> append '-v2'
        """
        parts = old_version.split(".")
        if parts and parts[-1].isdigit():
            parts[-1] = str(int(parts[-1]) + 1)
            return ".".join(parts)
        return f"{old_version}-v2"
