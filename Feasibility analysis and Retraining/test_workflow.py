"""
test_workflow.py
----------------
Automated Test Suite using Python's built-in `unittest` framework.
Tests feasibility constraints, catalog persistence, and handoff logic.
"""

import unittest
import os
import shutil
from feasibility import check_retraining_feasibility
from catalog import ModelCatalog
from handoff import decide_handoff


class TestRetrainingWorkflow(unittest.TestCase):

    def setUp(self):
        self.test_db = "./test_catalog.db"

    def tearDown(self):
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

    def test_feasibility_rejection_on_severe_drift(self):
        report = check_retraining_feasibility(
            new_data_size=1000, drift_severity=0.85, current_accuracy=0.60
        )
        self.assertFalse(report.feasible)
        self.assertTrue(any("Drift too severe" in r for r in report.reasons))

    def test_catalog_sqlite_versioning(self):
        catalog = ModelCatalog(db_path=self.test_db)
        entry1 = catalog.register_version("model-a", "./p1.pt", {}, 0.80, 0.78)
        entry2 = catalog.register_version("model-a", "./p2.pt", {}, 0.85, 0.83)
        self.assertEqual(entry1.version, 1)
        self.assertEqual(entry2.version, 2)

    def test_handoff_rejects_regression(self):
        decision = decide_handoff(
            feasible=True,
            model_path="./model.pt",
            new_metrics={"accuracy": 0.55, "f1": 0.50},
            previous_accuracy=0.65,
            previous_f1=0.60,
        )
        self.assertEqual(decision.action, "operator_review")
        self.assertIn("regressed", decision.reason)


if __name__ == "__main__":
    unittest.main()