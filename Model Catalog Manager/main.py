"""
main.py
-------
Runnable entry point for the Model Catalog Manager simulator.

Loads the sample incoming models (exact / similar / new) and runs each
through the ModelCatalogManager pipeline, printing a clear report of
what happened at each stage.

Run with:
    python main.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.catalog_manager import ModelCatalogManager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

CATALOG_PATH = os.path.join(DATA_DIR, "model_catalog.json")
HISTORY_PATH = os.path.join(DATA_DIR, "version_history.json")
QUEUE_PATH = os.path.join(DATA_DIR, "finetuning_queue.json")
SAMPLES_PATH = os.path.join(DATA_DIR, "incoming_samples.json")


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_case(manager, label, incoming_model):
    print_header(f"INCOMING MODEL CASE: {label}")
    print("Incoming metadata:")
    print(json.dumps(incoming_model, indent=2))

    result = manager.process_incoming_model(incoming_model)

    if result["status"] == "error":
        print(f"\n[ERROR] Stage='{result['stage']}' -> {result['message']}")
        return

    print(f"\nMatch type detected : {result['match_type']}")
    print(f"Action taken         : {result['action']}")
    print(f"Candidates considered: {result['candidates_considered']}")
    if result["matched_existing_model_id"]:
        print(f"Matched existing id  : {result['matched_existing_model_id']}")

    print("\nFinal stored model entry:")
    print(json.dumps(result["final_model"], indent=2))

    print("\nForwarded to Fine-tuning Engineer as job:")
    print(json.dumps(result["forwarded_job"], indent=2))


def main():
    manager = ModelCatalogManager(CATALOG_PATH, HISTORY_PATH, QUEUE_PATH)

    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    run_case(manager, "1) EXACT MATCH (same model_id, retrained)", samples["exact_match_example"])
    run_case(manager, "2) SIMILAR MATCH (new id, closely related name)", samples["similar_match_example"])
    run_case(manager, "3) NEW MODEL (no match in catalog)", samples["new_model_example"])

    print_header("PIPELINE COMPLETE")
    print(f"Updated catalog file : {CATALOG_PATH}")
    print(f"Version history file : {HISTORY_PATH}")
    print(f"Fine-tuning queue file: {QUEUE_PATH}")


if __name__ == "__main__":
    main()
