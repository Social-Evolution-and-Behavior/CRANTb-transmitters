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
        parallel=8,
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
    dataset = load_dataset(config, split="train")
    logging.info(f"Loaded training dataset with {len(dataset)} samples.")
    # seeded shuffling with a generator
    torch.manual_seed(config.seed)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=0,
    )
    model = load_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    # Get an example batch
    t0 = time.time()
    for epoch in range(config.train.epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader):
            x, y = batch
            logging.info(f"Input shape: {x.shape}, Label shape: {y.shape}")
            logging.info(f"Time taken to load a batch: {time.time() - t0:.2f} seconds")
            logging.info(f"Classes: {y}")
            t0 = time.time()
            # Loop
            optimizer.zero_grad()
            # Forward pass
            outputs = model(x)
            loss = loss_fn(outputs, y.long())
            epoch_loss += loss.item()
            # Backward pass
            loss.backward()
            optimizer.step()
            break
        logging.info(
            f"Epoch {epoch + 1}/{config.train.epochs}, Loss: {epoch_loss / len(dataloader)}"
        )
        # TODO save model checkpoints
        # TODO log metrics


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
