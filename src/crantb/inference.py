import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import torch
import numpy as np


def get_epoch_metrics(
    metrics_directory: str, epoch: int = None, metric: str = "balanced_accuracy"
):
    """
    Get the best epoch based on the validation loss from the metrics file.
    """
    metrics_path = Path(metrics_directory) / "training_metrics.csv"
    # Load the metrics
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics path {metrics_path} does not exist.")
    metrics = pd.read_csv(metrics_path)
    # Set the epoch as the index
    metrics.set_index("epoch", inplace=True)
    if epoch is None:
        # Select and print the epoch with the highest metric
        # Reversing the order to get the latest epoch with the best metric in case of ties
        epoch = metrics[metric][::-1].idxmax()
        logging.info(
            f"Selected epoch {epoch} with highest {metric}: {metrics[metric].max()}"
        )
    # Return metrics and the selected epoch
    return metrics.loc[epoch].to_dict(), epoch


def run_inference(model, dataloader, output, config):
    model.eval()
    all_predictions = []
    all_locations = []
    # Run inference and save the predictions
    for batch, locations in tqdm(dataloader, total=len(dataloader)):
        with torch.no_grad():
            outputs = model(batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=-1)
        all_predictions.extend(probabilities.cpu().numpy())
        all_locations.extend(locations.cpu().numpy())
    results = pd.DataFrame(all_predictions, columns=config.gt.neurotransmitters)
    results[["x", "y", "z"]] = all_locations
    # Save to output feather file
    results.to_feather(output)
