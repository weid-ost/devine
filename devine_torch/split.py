"""Dataset splitting strategies used during training."""

from typing import Any, Literal

import numpy as np
import xarray as xr
from torch.utils.data import Subset, random_split


class RandomSplit:
    """Randomly split a dataset according to provided ratios."""

    def __init__(self, split: list[float]):
        """Store split ratios in the order train, val, test."""
        self.split = split

    def split_dataset(self, dataset):
        """Return train and validation subsets."""

        train_dataset, val_dataset, _ = random_split(
            dataset,
            self.split,
        )
        return train_dataset, val_dataset


class CoordsSplit:
    """Split based on predefined coordinate indices stored in NetCDF files."""

    def __init__(self, train_coords_path, val_coords_path):
        """Load coordinate-based train/val indices from NetCDF paths."""
        ds_train = xr.load_dataset(train_coords_path)
        ds_val = xr.load_dataset(train_coords_path)

        self.ds_train = ds_train.rename({"sl": "slope"}).set_index(
            n=["xi", "slope", "sigma", "r"]
        )
        self.ds_val = ds_val.rename({"sl": "slope"}).set_index(
            n=["xi", "slope", "sigma", "r"]
        )

    def split_dataset(self, dataset):
        """Return train and validation subsets aligned to provided coordinates."""
        ds_tmp = dataset.feature_ds
        wind_dir_tmp = np.unique(ds_tmp.wind_dir)[0]
        n_wind_dirs = np.unique(ds_tmp.wind_dir).shape[0]

        ds_tmp = ds_tmp.where(ds_tmp["wind_dir"] == wind_dir_tmp, drop=True).drop_vars(
            "wind_dir"
        )
        ds_tmp = ds_tmp.set_index(n=["xi", "slope", "sigma", "r"])

        train_mask = ds_tmp.indexes["n"].get_indexer(self.ds_train.indexes["n"])
        val_mask = ds_tmp.indexes["n"].get_indexer(self.ds_val.indexes["n"])

        offsets = np.arange(n_wind_dirs) * ds_tmp.n.shape[0]

        train_indices = (train_mask[None, :] + offsets[:, None]).flatten()
        val_indices = (val_mask[None, :] + offsets[:, None]).flatten()

        return Subset(dataset, train_indices), Subset(dataset, val_indices)


split_registry: dict[Literal["random", "coords"], Any] = {
    "random": RandomSplit,
    "coords": CoordsSplit,
}
