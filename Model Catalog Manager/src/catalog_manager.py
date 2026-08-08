"""
catalog_manager.py
-------------------
ModelCatalogManager: the orchestrator that implements, in order, the
four responsibilities shown in the reference diagram:

    1. Search the Model Catalog for the incoming model.
    2. Check whether a similar or existing model is already registered.
    3. Replace the existing model entry if a match is found.
    4. Maintain model versions and forward the entry to the
       Fine-tuning Engineer.
"""

from .models import ModelMetadata, InvalidModelDataError
from .catalog_store import CatalogStore, CatalogStoreError
from .search_engine import SearchEngine
from .version_manager import VersionManager, VersionManagerError
from .finetuning_handoff import FineTuningHandoff, HandoffError


class ModelCatalogManager:
    def __init__(self, catalog_path: str, history_path: str, queue_path: str):
        self.store = CatalogStore(catalog_path)
        self.version_manager = VersionManager(history_path)
        self.handoff = FineTuningHandoff(queue_path)

    def process_incoming_model(self, raw_incoming: dict) -> dict:
        """
        Runs the full pipeline for one incoming model and returns a
        result summary dict describing what happened.
        """
        # --- Validate incoming metadata -----------------------------------
        try:
            incoming_obj = ModelMetadata.from_dict(raw_incoming)
        except InvalidModelDataError as e:
            return {"status": "error", "stage": "validation", "message": str(e)}

        incoming = incoming_obj.to_dict()

        # --- Step 1: Search the catalog ------------------------------------
        try:
            catalog_models = self.store.get_all_models()
        except CatalogStoreError as e:
            return {"status": "error", "stage": "catalog_read", "message": str(e)}

        engine = SearchEngine(catalog_models)

        try:
            candidates = engine.search(incoming)
        except Exception as e:
            return {"status": "error", "stage": "search", "message": str(e)}

        # --- Step 2: Determine whether it's exact / similar / new ---------
        match_type, matched_model = engine.determine_match(incoming)

        # --- Step 3 & 4: Replace / register, version, forward -------------
        try:
            if match_type in ("exact", "similar"):
                result_model, action = self._replace_existing(
                    catalog_models, matched_model, incoming, match_type
                )
            else:
                result_model, action = self._register_new(catalog_models, incoming)
        except (VersionManagerError, CatalogStoreError) as e:
            return {"status": "error", "stage": "persist", "message": str(e)}

        try:
            job = self.handoff.forward(result_model, action)
        except HandoffError as e:
            return {"status": "error", "stage": "handoff", "message": str(e)}

        return {
            "status": "success",
            "match_type": match_type,
            "action": action,
            "matched_existing_model_id": matched_model["model_id"] if matched_model else None,
            "final_model": result_model,
            "forwarded_job": job,
            "candidates_considered": len(candidates),
        }

    # ------------------------------------------------------------------
    def _replace_existing(self, catalog_models, matched_model, incoming, match_type):
        """Responsibility 3 + version bump (part of Responsibility 4)."""
        new_entry = dict(incoming)
        new_entry["model_id"] = matched_model["model_id"]  # keep catalog identity stable
        new_entry["version"] = self.version_manager.next_version(matched_model["version"])

        # Archive the version being replaced
        self.version_manager.archive_previous_version(matched_model, new_entry, match_type)

        # Replace in-place within the catalog list
        updated_models = []
        for m in catalog_models:
            if m["model_id"] == matched_model["model_id"]:
                updated_models.append(new_entry)
            else:
                updated_models.append(m)
        self.store.replace_all_models(updated_models)

        action = "updated_exact" if match_type == "exact" else "updated_similar"
        return new_entry, action

    def _register_new(self, catalog_models, incoming):
        """Responsibility 4 (registration branch when no match found)."""
        new_entry = dict(incoming)
        new_entry["version"] = new_entry.get("version") or "1.0"
        updated_models = catalog_models + [new_entry]
        self.store.replace_all_models(updated_models)
        return new_entry, "registered"
