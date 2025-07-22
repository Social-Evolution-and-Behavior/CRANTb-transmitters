import os

# Enable MPS fallback for unsupported operations - MUST be set before importing torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from crantb.data import CloudVolumeDataset, train_transform, test_transform
from crantb.split import split_data
from crantb.train import (
    train_one_epoch,
    validate_one_epoch,
    save_model_checkpoint,
    store_metrics,
    compute_val_metrics,
)
from monai.networks.nets import resnet
import logging
from pathlib import Path
from omegaconf import OmegaConf
import torch
import typer
import time
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
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
    for epoch in range(config.train.epochs):
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
            save_model_checkpoint(model, epoch, config)


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
