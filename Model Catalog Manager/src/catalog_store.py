"""
catalog_store.py
-----------------
A tiny JSON-file-backed "database" that simulates a persistent Model
Catalog. Responsible ONLY for reading and writing catalog data to disk.
"""

import json
import os


class CatalogStoreError(Exception):
    """Raised when the catalog file cannot be read or written."""
    pass


class CatalogStore:
    def __init__(self, catalog_path: str):
        self.catalog_path = catalog_path
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        os.makedirs(os.path.dirname(self.catalog_path), exist_ok=True)
        if not os.path.exists(self.catalog_path):
            with open(self.catalog_path, "w", encoding="utf-8") as f:
                json.dump({"models": []}, f, indent=2)

    def load(self) -> dict:
        try:
            with open(self.catalog_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {"models": []}
                return json.loads(content)
        except json.JSONDecodeError as e:
            raise CatalogStoreError(f"Catalog file is corrupted: {e}")
        except OSError as e:
            raise CatalogStoreError(f"Could not read catalog file: {e}")

    def save(self, catalog_data: dict):
        try:
            with open(self.catalog_path, "w", encoding="utf-8") as f:
                json.dump(catalog_data, f, indent=2)
        except OSError as e:
            raise CatalogStoreError(f"Could not write catalog file: {e}")

    def get_all_models(self) -> list:
        return self.load().get("models", [])

    def replace_all_models(self, models: list):
        self.save({"models": models})
