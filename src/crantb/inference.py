import pandas as pd
from pathlib import Path
import logging
import tqdm
import torch


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
    with open(output, "w") as f:
        # Prepare the output file
        f.write(
            "x,y,z,"
            + ",".join(config.gt.neurotransmitters)
            + ",predicted_neurotransmitter\n"
        )
        # Run inference and save the predictions
        for batch, locations in tqdm(dataloader, total=len(dataloader)):
            with torch.no_grad():
                outputs = model(batch)
                probabilities = torch.nn.functional.softmax(outputs, dim=-1)
                predictions = probabilities.argmax(dim=-1)

                # Write the predictions to the file
                for i in range(len(locations)):
                    line = (
                        f"{locations[i]['x']},{locations[i]['y']},{locations[i]['z']},"
                    )
                    line += ",".join(map(str, probabilities[i].cpu().numpy())) + ","
                    line += config.gt.neurotransmitters[predictions[i].item()] + "\n"
                    f.write(line)
