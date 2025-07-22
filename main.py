import os

# Enable MPS fallback for unsupported operations - MUST be set before importing torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from crantb.data import CloudVolumeDataset, train_transform, test_transform
from crantb.split import split_data
from monai.networks.nets import resnet
import logging
from pathlib import Path
from omegaconf import OmegaConf
import torch
import typer
import time
from tqdm import tqdm
from accelerate import Accelerator
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
import numpy as np
import pandas as pd


# The CLI
app = typer.Typer()


def load_config(cfg: str) -> OmegaConf:
    """
    Load configuration from a file.
    """
    config = OmegaConf.load(cfg)
    return config


def load_dataset(cfg: OmegaConf, split="train") -> CloudVolumeDataset:
    """
    Load the dataset based on the configuration.

    Uses the `data` part of the configuration file.
    """
    transform = train_transform() if split == "train" else test_transform()
    return CloudVolumeDataset(
        cloud_volume_path=cfg.data.container,
        metadata_path=cfg.gt[split],
        classes=cfg.gt.neurotransmitters,
        crop_size=cfg.train.input_shape,
        transform=transform,
        parallel=cfg.data.parallel,
        use_https=cfg.data.use_https,
        cache=cfg.data.cache,
        progress=cfg.data.progress,
    )


def load_model(cfg: OmegaConf) -> torch.nn.Module:
    """
    Load the model based on the configuration.

    Uses the `model` part of the configuration file.
    """
    model = resnet.ResNet(
        block="basic",
        layers=[3, 4, 6, 3],  # ResNet50 layer configuration
        block_inplanes=resnet.get_inplanes(),
        spatial_dims=len(cfg.data.voxel_size),
        n_input_channels=cfg.data.channels,
        num_classes=len(cfg.gt.neurotransmitters),
    )
    return model


@app.command()
def split(cfg: str = "config.yaml"):
    cfg = load_config(cfg)
    # Split the data
    train_gt, val_gt = split_data(
        base=cfg.gt.base,
        val_size=cfg.gt.val_size,
        random_state=cfg.seed,
        body_id=cfg.gt.body_id,
        nt_name=cfg.gt.nt_name,
        neurotransmitters=cfg.gt.neurotransmitters,
    )
    # Save the split data
    train_path = Path(cfg.gt.train)
    val_path = Path(cfg.gt.val)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    train_gt.to_feather(train_path)
    val_gt.to_feather(val_path)


@app.command()
def train(cfg: str = "config.yaml"):
    """
    Train the model based on the configuration.
    """
    # Initialize accelerator
    accelerator = Accelerator()

    config = load_config(cfg)
    dataset = load_dataset(config, split="train")
    val_dataset = load_dataset(config, split="val")

    # Log only on main process
    if accelerator.is_main_process:
        logging.info(f"Loaded training dataset with {len(dataset)} samples.")
        logging.info(f"Loaded validation dataset with {len(val_dataset)} samples.")
        logging.info(f"Using device: {accelerator.device}")
        logging.info(f"PyTorch version: {torch.__version__}")

    # seeded shuffling with a generator
    torch.manual_seed(config.seed)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.validate.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = load_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    # TODO add class weights to account for class imbalance
    loss_fn = torch.nn.CrossEntropyLoss(weight=None)
    val_loss_fn = torch.nn.CrossEntropyLoss(weight=None)  # No weight for validation

    # Prepare everything with accelerator
    model, optimizer, dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, dataloader, val_dataloader
    )

    # Training loop
    for epoch in range(config.train.epochs):
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

        # Log only on main process
        if accelerator.is_main_process:
            # Calculate metrics
            accuracy = accuracy_score(all_targets, all_predictions)
            balanced_accuracy = balanced_accuracy_score(all_targets, all_predictions)

            # Create confusion matrix with explicit class labels
            class_labels = list(range(len(config.gt.neurotransmitters)))
            conf_matrix = confusion_matrix(
                all_targets, all_predictions, labels=class_labels
            )

            # Create checkpoint and metrics directories
            checkpoint_dir = Path(config.train.base) / "checkpoints"
            metrics_dir = Path(config.train.base) / "metrics"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            metrics_dir.mkdir(parents=True, exist_ok=True)

            # Save scalar metrics to CSV
            metrics_data = {
                "epoch": epoch + 1,
                "train_loss": epoch_loss / len(dataloader),
                "val_loss": epoch_val_loss / len(val_dataloader),
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
            np.savetxt(conf_matrix_path, conf_matrix, fmt="%d")

            # Save model checkpoint
            torch.save(
                model.state_dict(),
                checkpoint_dir / f"model_epoch_{epoch + 1}.pth",
            )


def test(cfg: str):
    config = load_config(cfg)
    print("Testing with configuration:", config)
    # Here you would implement the testing logic using the config


def inference(cfg: str):
    config = load_config(cfg)
    print("Running inference with configuration:", config)
    # Here you would implement the inference logic using the config


if __name__ == "__main__":
    # Set logging level
    logging.basicConfig(level=logging.INFO)
    app()
