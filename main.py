import os

# Enable MPS fallback for unsupported operations - MUST be set before importing torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from accelerate import Accelerator
from accelerate.utils import set_seed
from crantb.data import CloudVolumeDataset, train_transform, test_transform
from crantb.split import split_data
from crantb.train import (
    train_one_epoch,
    validate_one_epoch,
    save_checkpoint,
    load_checkpoint,
    store_metrics,
    compute_val_metrics,
)
import logging
from monai.networks.nets import resnet
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from pathlib import Path
import torch
import typer
from typing import Optional, Annotated


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
def train(
    cfg: str = "config.yaml",
    epochs: Optional[int] = None,
    resume: Annotated[Optional[bool], typer.Option("--resume/--restart")] = True,
):
    """
    Train the model based on the configuration.
    """
    config = load_config(cfg)
    # Initialize accelerator
    set_seed(config.seed)
    accelerator = Accelerator()
    dataset = load_dataset(config, split="train")
    val_dataset = load_dataset(config, split="val")
    # Dataloaders
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
    # Initialize model, optimizer
    model = load_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)

    if not resume:
        # Delete the metrics folder and start fresh
        metrics_path = Path(config.train.base) / "metrics"
        if metrics_path.exists():
            for file in metrics_path.glob("*.txt"):
                file.unlink()
            for file in metrics_path.glob("*.csv"):
                file.unlink()
            logging.info(f"Deleted existing metrics in {metrics_path}")
        # Delete the checkpoints folder and start fresh
        checkpoints_path = Path(config.train.base) / "checkpoints"
        if checkpoints_path.exists():
            for file in checkpoints_path.glob("*.pth"):
                file.unlink()
            logging.info(f"Deleted existing checkpoints in {checkpoints_path}")

    # Load the latest checkpoint and
    # resume training.
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, config.train.base)

    # Prepare model and optimizer with accelerator
    model, optimizer, dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, dataloader, val_dataloader
    )

    # Losses
    class_weights = dataset.weights.to(accelerator.device)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )  # Use class weights to account for class imbalance
    val_loss_fn = torch.nn.CrossEntropyLoss(weight=None)  # No weight for validation

    # Training loop
    if epochs is None:
        epochs = config.train.epochs

    for epoch in range(start_epoch, epochs):
        epoch_loss = train_one_epoch(
            epoch, model, dataloader, optimizer, loss_fn, accelerator
        )
        epoch_val_loss, predictions, targets = validate_one_epoch(
            epoch, model, val_dataloader, val_loss_fn, accelerator, config
        )
        if accelerator.is_main_process:
            accuracy, balanced_accuracy, confusion_matrix = compute_val_metrics(
                predictions, targets, config
            )
            store_metrics(
                epoch,
                epoch_loss=epoch_loss,
                epoch_val_loss=epoch_val_loss,
                accuracy=accuracy,
                balanced_accuracy=balanced_accuracy,
                confusion_matrix=confusion_matrix,
                config=config,
            )
            save_checkpoint(model, optimizer, epoch, config)


@app.command()
def report(cfg: str = "config.yaml", epoch: int = None, metric="balanced_accuracy"):
    """
    Report the results of the training.

    This reads through the metrics files and prints the results.
    """
    config = load_config(cfg)
    # Load the metrics
    metrics_path = Path(config.train.base) / "metrics"
    if not metrics_path.exists():
        logging.error(f"Metrics path {metrics_path} does not exist.")
        return
    metrics = pd.read_csv(metrics_path / "training_metrics.csv")
    # Set the epoch as the index
    metrics.set_index("epoch", inplace=True)
    if epoch is None:
        # Select and print the epoch with the highest metric
        epoch = metrics[metric].idxmax()
        print(f"Selected epoch {epoch} with highest {metric}: {metrics[metric].max()}")

    print(f"Results for epoch {epoch}:")
    for name, value in metrics.loc[epoch].to_dict().items():
        print(f"\t- {name}: {value}")
    # Get the confusion matrix from the file
    confusion_matrix = np.loadtxt(metrics_path / f"confusion_matrix_epoch_{epoch}.txt")
    print()
    print("Confusion Matrix:")
    print(confusion_matrix)


def inference(cfg: str):
    config = load_config(cfg)
    print("Running inference with configuration:", config)
    # Here you would implement the inference logic using the config


if __name__ == "__main__":
    # Set logging level
    logging.basicConfig(level=logging.INFO)
    app()
