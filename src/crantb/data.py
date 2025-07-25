import cloudvolume
import pandas as pd
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    ScaleIntensity,
    RandRotate90,
    RandAxisFlip,
    RandScaleIntensityFixedMean,
    RandShiftIntensity,
)
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Union


def train_transform():
    transform = Compose(
        [
            EnsureChannelFirst(channel_dim=-1),
            ScaleIntensity(minv=-1, maxv=1),
            # Augmentations
            RandAxisFlip(prob=0.5),
            RandRotate90(prob=0.5, spatial_axes=(0, 1)),
            RandScaleIntensityFixedMean(prob=0.5, factors=0.2),
            RandShiftIntensity(prob=0.5, offsets=0.2),
        ]
    )
    return transform


def test_transform():
    transform = Compose(
        [
            EnsureChannelFirst(channel_dim=-1),
            ScaleIntensity(minv=-1, maxv=1),
        ]
    )
    return transform


def compute_class_weights(classes, num_classes):
    """
    Compute class weights based on the frequency of each class.
    """
    class_counts = np.bincount(classes, minlength=num_classes)
    # Handle classes that don't appear in this split
    class_counts = np.where(class_counts == 0, 1, class_counts)
    total_count = class_counts.sum()
    class_weights = total_count / (num_classes * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    assert (
        len(class_weights) == num_classes
    ), f"Class weights length {len(class_weights)} does not match number of classes {num_classes}"
    return class_weights


class CloudVolumeDataset(Dataset):
    """
    Dataset that allows access to specific locations in a cloud volume array.
    """

    def __init__(
        self,
        cloud_volume_path,
        metadata_dataframe: pd.DataFrame,
        classes=None,
        crop_size=(64, 64, 64),
        transform=None,
        cache=None,
        use_https=True,
        parallel=True,
        progress=False,
        inference=False,
        sample_weights_column: str = None,
        class_weights: Union[torch.Tensor, bool] = True,
    ):
        """
        Args:
            cloud_volume_path (str): Path to the cloud volume.
            metadata_dataframe (pd.DataFrame): DataFrame containing metadata with 'x', 'y', 'z' columns for locations.
            classes (list, optional): List of class names. If None, will use all available classes in the metadata.
            crop_size (tuple): Size of the crop around each location.
            transform (callable, optional): Transform to apply to the cropped volume.
            cache (bool, optional): Whether to cache the cloud volume.
            use_https (bool, optional): Whether to use HTTPS for the cloud volume.
            parallel (bool, optional): Whether to use parallel loading for data loading.
            progress (bool, optional): Whether to show progress for data loading.
            inference (bool): If True, the dataset is used for inference and does not require targets
            sample_weights_column (str, optional): Column name in metadata for sample weights.
            class_weights (Any[torch.Tensor, None, bool], optional): Precomputed class weights as a torch.Tensor,
                False to disable class weights, or True to compute class weights from targets.
                Defaults to True, which computes class weights from targets.
        """
        super().__init__()
        self.inference = inference
        self.locations, self.targets, self.class_names = self._read_metadata(
            metadata_dataframe, classes
        )
        if class_weights is True:
            # Compute class weights from targets
            self.class_weights = compute_class_weights(
                self.targets, len(self.class_names)
            )
        elif isinstance(class_weights, torch.Tensor):
            # Use provided class weights
            self.class_weights = class_weights
        elif class_weights is False:
            # Disable class weights --> set them all to one
            self.class_weights = torch.ones(len(self.class_names), dtype=torch.float32)
        if sample_weights_column:
            if sample_weights_column not in metadata_dataframe.columns:
                raise ValueError(
                    f"Column '{sample_weights_column}' not found in metadata DataFrame."
                )
            self.sample_weights = metadata_dataframe[sample_weights_column].values
        else:
            # Sample weights for each location, if provided, else set to ones
            self.sample_weights = torch.ones(len(self.locations), dtype=torch.float32)
        self.crop_size = crop_size  # Size around each location to crop
        self.transform = transform
        # Setup for the cloud volume
        self.cloud_volume_path = cloud_volume_path
        self.cloud_volume = cloudvolume.CloudVolume(
            self.cloud_volume_path,
            cache=cache,
            use_https=use_https,
            parallel=parallel,
            progress=progress,
        )

    def _read_metadata(self, metadata, classes=None):
        """
        Reads metadata from a Feather file and extracts locations and classes.
        """
        locations = metadata[["x", "y", "z"]].values
        if self.inference:
            try:
                class_names = {i: name for i, name in enumerate(sorted(classes))}
            except TypeError:
                # Classes need to be provided in inference mode
                raise ValueError(
                    "Classes must be provided in inference mode. "
                    "Please provide a list of class names."
                )
            # For inference, we only need the locations
            return locations, None, class_names
        # Get all available class names from the metadata column, even if not present in this split
        available_classes = (
            metadata["neurotransmitter"].astype("category").cat.categories
        )
        if classes is None:
            classes = available_classes
        # Only use provided classes, but ensure all possible classes are counted
        class_names = pd.Categorical(classes, categories=classes).categories
        # turn into a dictionary from index to name
        class_names = {i: name for i, name in enumerate(class_names)}
        # Convert the classes to numerical values
        targets = (
            metadata["neurotransmitter"].astype("category").cat.codes.values
        )  # Neurotransmitters to numerical
        return locations, targets, class_names

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, idx):
        if idx >= len(self.locations):
            raise IndexError("Index out of bounds for dataset.")

        loc = self.locations[idx]
        x, y, z = loc

        # Calculate the crop bounds
        half_crop = tuple(s // 2 for s in self.crop_size)
        start = (x - half_crop[0], y - half_crop[1], z - half_crop[2])
        end = (x + half_crop[0], y + half_crop[1], z + half_crop[2])

        # Fetch the cropped volume
        cropped_volume = self.cloud_volume[
            start[0] : end[0], start[1] : end[1], start[2] : end[2]
        ]

        if self.transform:
            cropped_volume = self.transform(cropped_volume)

        if self.inference:
            # For inference, we return the volume and the position
            return cropped_volume, loc

        return cropped_volume, self.targets[idx]
