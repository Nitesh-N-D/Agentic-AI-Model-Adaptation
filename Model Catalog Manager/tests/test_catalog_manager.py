"""
test_catalog_manager.py
------------------------
Test cases for:
  1. Existing model  (exact match -> replace)
  2. Similar model   (fuzzy match -> replace)
  3. New model       (no match -> register)

Each test runs in an isolated temp directory so it never touches the
real data/ files used by main.py.

Run with:
    python -m unittest discover tests
"""

import unittest
import tempfile
import shutil
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.catalog_manager import ModelCatalogManager


SEED_CATALOG = {
    "models": [
        {
            "model_id": "mdl-1001",
            "name": "SentimentBERT",
            "version": "1.0",
            "task_type": "text-classification",
            "architecture": "transformer",
            "framework": "pytorch",
            "domain": "general",
            "size_params": "110M",
            "accuracy": 0.91,
            "source": "internal-team-nlp",
            "tags": ["sentiment", "bert"],
            "submitted_at": "2025-11-02T09:15:00",
        }
    ]
}


class TestModelCatalogManager(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.catalog_path = os.path.join(self.tmp_dir, "model_catalog.json")
        self.history_path = os.path.join(self.tmp_dir, "version_history.json")
        self.queue_path = os.path.join(self.tmp_dir, "finetuning_queue.json")

        with open(self.catalog_path, "w", encoding="utf-8") as f:
            json.dump(SEED_CATALOG, f)

        self.manager = ModelCatalogManager(self.catalog_path, self.history_path, self.queue_path)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    # ------------------------------------------------------------------
    def test_existing_model_exact_match_replaces_entry(self):
        incoming = {
            "model_id": "mdl-1001",
            "name": "SentimentBERT",
            "version": "1.0",
            "task_type": "text-classification",
            "architecture": "transformer",
            "framework": "pytorch",
            "accuracy": 0.95,
        }
        result = self.manager.process_incoming_model(incoming)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["match_type"], "exact")
        self.assertEqual(result["action"], "updated_exact")
        self.assertEqual(result["final_model"]["model_id"], "mdl-1001")
        self.assertEqual(result["final_model"]["version"], "1.1")  # version bumped
        self.assertEqual(result["final_model"]["accuracy"], 0.95)

        # Catalog should still contain exactly one entry for this model
        with open(self.catalog_path) as f:
            catalog = json.load(f)
        self.assertEqual(len(catalog["models"]), 1)
        self.assertEqual(catalog["models"][0]["version"], "1.1")

        # History should have exactly one archived version
        with open(self.history_path) as f:
            history = json.load(f)
        self.assertEqual(len(history["history"]), 1)
        self.assertEqual(history["history"][0]["archived_model"]["version"], "1.0")

        # Job should be forwarded to the fine-tuning queue
        with open(self.queue_path) as f:
            queue = json.load(f)
        self.assertEqual(len(queue["queue"]), 1)
        self.assertEqual(queue["queue"][0]["action"], "updated_exact")

    def test_similar_model_replaces_closest_entry(self):
        incoming = {
            "model_id": "mdl-9999",              # different id
            "name": "SentimentBERT-v2",          # similar name
            "version": "1.0",
            "task_type": "text-classification",
            "architecture": "transformer",
            "framework": "pytorch",
        }
        result = self.manager.process_incoming_model(incoming)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["match_type"], "similar")
        self.assertEqual(result["action"], "updated_similar")
        # Identity of original catalog entry is preserved (not the new incoming id)
        self.assertEqual(result["final_model"]["model_id"], "mdl-1001")
        self.assertEqual(result["final_model"]["name"], "SentimentBERT-v2")

        with open(self.catalog_path) as f:
            catalog = json.load(f)
        self.assertEqual(len(catalog["models"]), 1)  # replaced, not duplicated

    def test_new_model_gets_registered(self):
        incoming = {
            "model_id": "mdl-3090",
            "name": "SpeechToTextWav2Vec",
            "version": "1.0",
            "task_type": "speech-recognition",
            "architecture": "transformer",
            "framework": "pytorch",
        }
        result = self.manager.process_incoming_model(incoming)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["match_type"], "new")
        self.assertEqual(result["action"], "registered")
        self.assertIsNone(result["matched_existing_model_id"])

        with open(self.catalog_path) as f:
            catalog = json.load(f)
        self.assertEqual(len(catalog["models"]), 2)  # added, original untouched
        ids = [m["model_id"] for m in catalog["models"]]
        self.assertIn("mdl-3090", ids)
        self.assertIn("mdl-1001", ids)

        with open(self.history_path) as f:
            history = json.load(f)
        self.assertEqual(len(history["history"]), 0)  # nothing archived for a new model

    # ------------------------------------------------------------------
    def test_invalid_incoming_model_is_rejected(self):
        incoming = {"name": "MissingFieldsModel"}  # missing required fields
        result = self.manager.process_incoming_model(incoming)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "validation")


if __name__ == "__main__":
    unittest.main()
