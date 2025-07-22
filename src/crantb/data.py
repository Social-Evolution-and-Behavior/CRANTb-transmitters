import cloudvolume
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    ScaleIntensity,
    RandRotate90,
    RandAxisFlip,
    RandScaleIntensityFixedMean,
    RandShiftIntensity,
)


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
    total_count = class_counts.sum()
    class_weights = total_count / (num_classes * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    return class_weights


class CloudVolumeDataset(Dataset):
    """
    Dataset that allows access to specific locations in a cloud volume array.
    """

    def __init__(
        self,
        cloud_volume_path,
        metadata_path,
        classes=None,
        crop_size=(64, 64, 64),
        transform=None,
        cache=None,
        use_https=True,
        parallel=True,
        progress=False,
    ):
        super().__init__()
        self.locations, self.classes, self.class_names = self._read_metadata(
            metadata_path, classes
        )
        self.weights = compute_class_weights(self.classes, len(self.class_names))
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

    def _read_metadata(self, metadata_path, classes=None):
        """
        Reads metadata from a Feather file and extracts locations and classes.
        """
        metadata = pd.read_feather(metadata_path)
        locations = metadata[["x", "y", "z"]].values
        if classes is not None:
            # If classes are provided, filter the metadata
            metadata = metadata[metadata["neurotransmitter"].isin(classes)]
        classes = metadata["neurotransmitter"]
        # Get the names of the classes
        class_names = classes.astype("category").cat.categories
        # turn into a dictionary from index to name
        class_names = {i: name for i, name in enumerate(class_names)}
        # Convert the classes to numerical values
        classes = classes.astype(
            "category"
        ).cat.codes.values  # Neurotransmitters to numerical
        return locations, classes, class_names

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

        return cropped_volume, self.classes[idx]
