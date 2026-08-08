"""
fine_tuning_engineer.py
------------------------
Simulates the "Fine-tuning Engineer" side of the handoff.

The Model Catalog Manager never calls this file directly -- it only
writes jobs onto data/finetuning_queue.json (see
src/finetuning_handoff.py). This script represents an independent
consumer/service that polls that same queue file, "picks up" pending
jobs, and marks them as picked up. This mirrors a real system where
the Catalog Manager and the Fine-tuning Engineer are separate
services/roles connected only through the shared queue.

Run with:
    python fine_tuning_engineer.py
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUEUE_PATH = os.path.join(BASE_DIR, "data", "finetuning_queue.json")


def main():
    if not os.path.exists(QUEUE_PATH):
        print("No fine-tuning queue found yet. Run main.py first.")
        return

    with open(QUEUE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    pending = [j for j in data["queue"] if j["status"] == "pending_fine_tuning"]

    if not pending:
        print("No pending fine-tuning jobs.")
        return

    print(f"Fine-tuning Engineer picked up {len(pending)} job(s):\n")
    for job in pending:
        model = job["model"]
        print(f"- {job['job_id']} | action={job['action']} | "
              f"model={model['name']} v{model['version']} ({model['model_id']})")
        job["status"] = "picked_up_by_fine_tuning_engineer"

    with open(QUEUE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("\nQueue file updated -- jobs marked as picked up.")


if __name__ == "__main__":
    main()
