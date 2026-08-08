"""
finetuning_handoff.py
----------------------
Responsibility 4 (part B): Forward the entry to the Fine-tuning Engineer.

This simulates a real handoff (e.g. a message queue / inbox file) rather
than just printing. The Model Catalog Manager writes each forwarded
model as a job onto a JSON-backed queue file. A separate
"Fine-tuning Engineer" consumer (see fine_tuning_engineer.py at the
project root) can independently read and process that queue -- this
mirrors how two separate roles/services would hand off work in a real
pipeline.
"""

import json
import os
import datetime


class HandoffError(Exception):
    pass


class FineTuningHandoff:
    def __init__(self, queue_path: str):
        self.queue_path = queue_path
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        os.makedirs(os.path.dirname(self.queue_path), exist_ok=True)
        if not os.path.exists(self.queue_path):
            with open(self.queue_path, "w", encoding="utf-8") as f:
                json.dump({"queue": []}, f, indent=2)

    def forward(self, model: dict, action: str) -> dict:
        """
        Places the finalized model entry onto the fine-tuning queue.
        `action` records why it was forwarded: "registered", "updated_exact",
        "updated_similar".
        Returns the job record that was queued.
        """
        try:
            with open(self.queue_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                data = json.loads(content) if content else {"queue": []}
        except (OSError, json.JSONDecodeError) as e:
            raise HandoffError(f"Could not read fine-tuning queue: {e}")

        job = {
            "job_id": f"job-{len(data['queue']) + 1:04d}",
            "model": model,
            "action": action,
            "forwarded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "pending_fine_tuning",
        }
        data["queue"].append(job)

        try:
            with open(self.queue_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            raise HandoffError(f"Could not write to fine-tuning queue: {e}")

        return job
