"""
main.py
-------
Pipeline Orchestration Script.
Enhancements: Command-line arguments (`argparse`), complete logging setup,
and full exception handling.
"""

import argparse
import logging
import sys
import torch
from torch.utils.data import Dataset

from feasibility import check_retraining_feasibility
from retrain import retrain_model
from catalog import ModelCatalog
from handoff import decide_handoff

# Configure Dual-Logging (Console + File)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("member4_workflow.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("WorkflowOrchestrator")


class SyntheticDataset(Dataset):
    """Synthetic dataset generator for testing and demonstration."""
    def __init__(self, n=1200, input_dim=10, num_classes=3):
        self.x = torch.randn(n, input_dim)
        self.y = torch.randint(0, num_classes, (n,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def run_pipeline(
    model_id: str,
    new_data_size: int,
    drift_severity: float,
    previous_accuracy: float,
    previous_f1: float,
):
    logger.info("Starting Retraining Pipeline for Model: %s", model_id)
    catalog = ModelCatalog("./model_catalog.db")

    try:
        # Step 1: Feasibility Evaluation
        logger.info("Step 1: Running Feasibility Check...")
        feasibility_report = check_retraining_feasibility(
            new_data_size=new_data_size,
            drift_severity=drift_severity,
            current_accuracy=previous_accuracy,
        )
        print("\n" + feasibility_report.summary() + "\n")

        if not feasibility_report.feasible:
            logger.warning("Pipeline halted: Feasibility failed. Routing to Operator.")
            return

        # Step 2: Fine-Tuning
        logger.info("Step 2: Commencing Model Fine-Tuning...")
        dataset = SyntheticDataset(n=new_data_size)
        retrain_res = retrain_model(
            existing_model_path="./current_model.pt",
            dataset=dataset,
            input_dim=10,
            hidden_dim=32,
            num_classes=3,
            epochs=3,
        )
        logger.info("Retraining completed. Metrics: %s", retrain_res.metrics)

        # Step 3: Cataloging
        logger.info("Step 3: Registering Model in Catalog...")
        entry = catalog.register_version(
            model_id=model_id,
            model_path=retrain_res.model_path,
            trigger_metrics=feasibility_report.metrics,
            validation_accuracy=retrain_res.metrics["accuracy"],
            validation_f1=retrain_res.metrics["f1"],
            status="pending_validation",
        )

        # Step 4: Handoff Evaluation
        logger.info("Step 4: Deciding Handoff Destination...")
        handoff = decide_handoff(
            feasible=feasibility_report.feasible,
            model_path=retrain_res.model_path,
            new_metrics=retrain_res.metrics,
            previous_accuracy=previous_accuracy,
            previous_f1=previous_f1,
        )

        final_status = "deployed" if handoff.action == "deploy_to_team3" else "operator_review"
        catalog.update_status(model_id, entry.version, final_status)
        logger.info("Pipeline Execution Complete. Final Action: %s", handoff.action)

    except Exception as e:
        logger.critical("Pipeline execution crashed: %s", e, exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Member 4 Retraining Workflow Execution")
    parser.add_argument("--model-id", type=str, default="airan-traffic-classifier")
    parser.add_argument("--data-size", type=int, default=1200)
    parser.add_argument("--drift", type=float, default=0.35)
    parser.add_argument("--prev-acc", type=float, default=0.62)
    parser.add_argument("--prev-f1", type=float, default=0.60)

    args = parser.parse_args()
    run_pipeline(
        model_id=args.model_id,
        new_data_size=args.data_size,
        drift_severity=args.drift,
        previous_accuracy=args.prev_acc,
        previous_f1=args.prev_f1,
    )
