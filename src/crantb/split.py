import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from omegaconf import DictConfig
import yaml


def split_data(
    base: str = "",
    val_size: float = 0.05,
    random_state: int = 12345,
    body_id: str = "neuron_id",
    nt_name: str = "neurotransmitter",
    neurotransmitters: list = None,
):
    """
    Split the ground truth data into training and validation sets, stratified by neurotransmitter type.

    Parameters
    ----------
    base : str
        Location of the ground truth data. Expected to be a feather file.
    val_size : float
        Proportion of the data to include in the validation set.
    random_state : int
        Random seed for reproducibility.
    body_id: str
        Name of the column containing the body IDs.
    nt_name: str
        Name of the column containing the neurotransmitter names.
    """
    gt = pd.read_feather(base)
    # Only keep the relevant neurotransmitters if specified
    if neurotransmitters is not None:
        gt = gt[gt[nt_name].isin(neurotransmitters)].copy()
    # Make sure that the required columns are present
    assert all([col in gt.columns for col in [body_id, nt_name]])
    # Print some basic information
    logging.info(f"Total number of synapses: {len(gt)}")
    logging.info(gt[nt_name].value_counts())
    # Get the body IDs
    body_ids = gt[[body_id, nt_name]].sort_values(body_id).drop_duplicates()

    # Splitting the body IDs into training and validation sets, stratified by neurotransmitter type
    train_body, val_body = train_test_split(
        body_ids,
        test_size=val_size,
        stratify=body_ids[nt_name],
        random_state=random_state,
    )
    # Ensure that there is no overlap between the training and validation sets
    # Only allow an overlap can only exist if the body IDs have multiple neurotransmitters
    overlap = set(train_body[body_id]) & set(val_body[body_id])
    try:
        assert overlap == set()
    except AssertionError as e:
        # Drop the overlapping body IDs from both sets
        logging.warning(f"Overlapping body IDs found: {overlap}")
        train_body = train_body[~train_body[body_id].isin(overlap)]
        val_body = val_body[~val_body[body_id].isin(overlap)]
        logging.warning(
            f"Overlapping body IDs removed. New training set size: {len(train_body)}"
        )
        logging.warning(f"New validation set size: {len(val_body)}")

    # Split the ground truth data into training and validation sets
    train_gt = gt[gt[body_id].isin(train_body[body_id])]
    val_gt = gt[gt[body_id].isin(val_body[body_id])]
    # print some basic information about the split
    logging.info(f"Training set size: {len(train_gt)}")
    logging.info(train_gt[nt_name].value_counts())
    logging.info(f"Validation set size: {len(val_gt)}")
    logging.info(val_gt[nt_name].value_counts())
    return train_gt, val_gt
