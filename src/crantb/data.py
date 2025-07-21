import cloudvolume
import pandas as pd
from torch.utils.data import Dataset


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
        use_https=True,
        cache=True,
        progress=False,
    ):
        super().__init__()
        self.cloud_volume = cloudvolume.CloudVolume(
            cloud_volume_path, use_https=use_https, cache=cache, progress=progress
        )
        self.locations, self.classes, self.class_names = self._read_metadata(
            metadata_path, classes
        )
        self.crop_size = crop_size  # Size around each location to crop

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

        return cropped_volume, self.classes[idx]
