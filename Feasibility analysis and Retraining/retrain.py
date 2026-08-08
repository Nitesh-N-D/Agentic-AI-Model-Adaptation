"""
retrain.py
----------
Step 2: Retrain the Model.
Enhancements: Multiclass evaluation metrics (Accuracy, F1-Score, Precision, Recall),
tensor dimension validation, and structured error handling.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


@dataclass
class RetrainResult:
    model_path: str
    final_train_loss: float
    metrics: Dict[str, float]  # val_accuracy, val_f1, val_precision, val_recall
    epochs_run: int


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_one_epoch(model, loader, optimizer, loss_fn, device, expected_dim) -> float:
    model.train()
    running_loss = 0.0
    for x_batch, y_batch in loader:
        if x_batch.shape[1] != expected_dim:
            raise ValueError(f"Expected feature dimension {expected_dim}, but got {x_batch.shape[1]}")

        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_batch.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def _evaluate_all_metrics(model, loader, device) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_targets = []

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        preds = model(x_batch).argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())

    if not all_targets:
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    acc = float((torch.tensor(all_preds) == torch.tensor(all_targets)).float().mean().item())
    f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))
    prec = float(precision_score(all_targets, all_preds, average="weighted", zero_division=0))
    rec = float(recall_score(all_targets, all_preds, average="weighted", zero_division=0))

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
    }


def retrain_model(
        existing_model_path: str,
        dataset: Dataset,
        *,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        val_split: float = 0.2,
        output_dir: str = "./retrained_models",
        device: str = "cpu",
) -> RetrainResult:
    """Fine-tunes the model on new data and returns evaluation metrics."""

    if len(dataset) == 0:
        raise ValueError("Cannot retrain on an empty dataset.")

    device_obj = torch.device(device)
    model = SimpleClassifier(input_dim, hidden_dim, num_classes).to(device_obj)

    ckpt_path = Path(existing_model_path)
    if ckpt_path.exists():
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device_obj))
            logger.info("Successfully loaded existing weights from %s", ckpt_path)
        except Exception as e:
            logger.warning("Failed to load checkpoint at %s (%s). Starting fresh.", ckpt_path, e)
    else:
        logger.info("No checkpoint found at %s. Initializing model from scratch.", ckpt_path)

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    final_loss = 0.0
    for epoch in range(1, epochs + 1):
        final_loss = _train_one_epoch(model, train_loader, optimizer, loss_fn, device_obj, input_dim)
        metrics = _evaluate_all_metrics(model, val_loader, device_obj)
        logger.info("Epoch %d/%d - Train Loss: %.4f - Val Acc: %.4f - Val F1: %.4f",
                    epoch, epochs, final_loss, metrics["accuracy"], metrics["f1"])

    eval_metrics = _evaluate_all_metrics(model, val_loader, device_obj)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "model_retrained.pt"
    torch.save(model.state_dict(), out_path)
    logger.info("Saved retrained checkpoint to %s", out_path)

    return RetrainResult(
        model_path=str(out_path),
        final_train_loss=final_loss,
        metrics=eval_metrics,
        epochs_run=epochs,
    )