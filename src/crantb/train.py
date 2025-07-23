import logging
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
import torch
from tqdm import tqdm


def store_metrics(
    epoch: int,
    epoch_loss: float,
    epoch_val_loss: float,
    accuracy: float,
    balanced_accuracy: float,
    confusion_matrix: np.ndarray,
    config: OmegaConf,
):
    metrics_dir = Path(config.train.base) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Save scalar metrics to CSV
    metrics_data = {
        "epoch": epoch + 1,
        "train_loss": epoch_loss,
        "val_loss": epoch_val_loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
    }

    metrics_csv_path = metrics_dir / "training_metrics.csv"

    # Create CSV with headers if it doesn't exist, otherwise append
    if not metrics_csv_path.exists():
        metrics_df = pd.DataFrame([metrics_data])
        metrics_df.to_csv(metrics_csv_path, index=False)
    else:
        metrics_df = pd.DataFrame([metrics_data])
        metrics_df.to_csv(metrics_csv_path, mode="a", header=False, index=False)

    # Save confusion matrix as raw matrix using np.savetxt
    conf_matrix_path = metrics_dir / f"confusion_matrix_epoch_{epoch + 1}.txt"
    np.savetxt(conf_matrix_path, confusion_matrix, fmt="%d")


def validate_one_epoch(epoch, model, val_dataloader, val_loss_fn, accelerator, config):
    # Validation phase
    model.eval()
    epoch_val_loss = 0.0
    all_predictions = []
    all_targets = []

    for batch in tqdm(
        val_dataloader,
        disable=not accelerator.is_main_process,
        desc=f"Validation Epoch {epoch+1}",
    ):
        x, y = batch
        with torch.no_grad():
            outputs = model(x)
            val_loss = val_loss_fn(outputs, y.long())
            epoch_val_loss += val_loss.item()

            # Collect predictions and targets for metrics
            y_pred = outputs.argmax(dim=-1)
            all_predictions.extend(y_pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    return epoch_val_loss / len(val_dataloader), all_predictions, all_targets


def compute_val_metrics(all_predictions: list, all_targets: list, config: OmegaConf):
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    balanced_accuracy = balanced_accuracy_score(all_targets, all_predictions)

    # Create confusion matrix with explicit class labels
    class_labels = list(range(len(config.gt.neurotransmitters)))
    conf_matrix = confusion_matrix(all_targets, all_predictions, labels=class_labels)
    return accuracy, balanced_accuracy, conf_matrix


def save_checkpoint(model, optimizer, epoch, config):
    """Save the model checkpoint after each epoch."""
    # Create checkpoint and metrics directories
    checkpoint_dir = Path(config.train.base) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model and optimizer checkpoint
    torch.save(
        {
            "epoch": epoch + 1,  # Store epoch as 1-based index
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_dir / f"model_epoch_{epoch + 1}.pth",
    )


def load_checkpoint(model, optimizer, base_directory):
    """
    Find and load the latest checkpoint.

    It will be under base_directory/checkpoints/ and will have the format
    model_epoch_X.pth where X is the epoch number.

    Also returns the epoch number of the loaded checkpoint.
    """
    checkpoint_dir = Path(base_directory) / "checkpoints"
    checkpoints = list(checkpoint_dir.glob("model_epoch_*.pth"))
    if not checkpoints:
        return model, optimizer, 0  # No checkpoint found, return model and epoch 0

    # Sort checkpoints by epoch number
    checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1]))
    latest_checkpoint = checkpoints[-1]

    # Load the model state
    checkpoint = torch.load(latest_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = int(latest_checkpoint.stem.split("_")[-1])  # Convert to zero-based index
    return model, optimizer, epoch


def train_one_epoch(epoch, model, dataloader, optimizer, loss_fn, accelerator):
    # Training phase
    model.train()
    epoch_loss = 0.0
    for batch in tqdm(
        dataloader,
        disable=not accelerator.is_main_process,
        desc=f"Training Epoch {epoch+1}",
    ):
        x, y = batch
        # Training Loop
        optimizer.zero_grad()
        # Forward pass
        outputs = model(x)
        loss = loss_fn(outputs, y.long())
        epoch_loss += loss.item()
        # Backward pass with accelerator
        accelerator.backward(loss)
        optimizer.step()
    return epoch_loss / len(dataloader)
