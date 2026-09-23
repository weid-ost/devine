"""Dataset utilities and dataloaders for WindMet training."""

import random
from pathlib import Path
from typing import Callable, Hashable, Literal, Protocol, Sequence

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from devine_torch.config import (
    Settings,
    TrainDataSettings,
    TrainSettings,
)

from ..split import split_registry


class Transform(Protocol):
    """Protocol for transforms that normalise or augment tensors."""

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor: ...

    @classmethod
    def load(cls, file_path: str | Path) -> "Transform": ...

    def save(self, file_path: str | Path): ...

    def state_dict(self) -> dict: ...


class NormaliseTransform:
    """Channel-wise normalisation transform using stored mean and std."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = mean
        self.std = std
        self.transform: Callable = v2.Normalize(mean=mean, std=std)

    def fit_transform(self, x: torch.Tensor):
        return self.transform(x)

    @classmethod
    def load(cls, file_path: str | Path) -> "NormaliseTransform":
        checkpoint = torch.load(file_path)

        return cls(checkpoint["mean"], checkpoint["std"])

    def save(self, file_path: str | Path):
        torch.save(self.state_dict(), file_path)

    def state_dict(self) -> dict[str, Sequence[float]]:
        return {"mean": self.mean, "std": self.std}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class ArpsDataset(Dataset):
    """Dataset wrapper for paired feature/target NetCDF files."""

    def __init__(
        self,
        feature_file: str | Path,
        target_file: str | Path,
        transform: Transform | None = None,
    ):
        self.feature_ds = xr.open_dataset(feature_file)

        self.target_ds = xr.open_dataset(target_file)

        self.transform = transform

        self._data: torch.Tensor | None = None
        self._labels: torch.Tensor | None = None

        self._init_data()

    def _init_data(self):
        self._data = torch.from_numpy(
            self.feature_ds.to_dataarray()
            .transpose("n", "variable", "nrows", "ncols")
            .values
        ).to(torch.float32)

        self._labels = torch.from_numpy(
            self.target_ds.to_dataarray()
            .transpose("n", "variable", "nrows", "ncols")
            .values
        ).to(torch.float32)

        self._metadata: dict[Hashable, int | list[str] | str] = dict(
            self.feature_ds.sizes
        )
        self._metadata["features"] = list(self.feature_ds.data_vars.keys())
        self._metadata["labels"] = list(self.target_ds.data_vars.keys())

    def __repr__(self) -> str:
        if self.transform is not None:
            self._metadata["transform"] = str(self.transform)
        return str(self._metadata)

    def __str__(self) -> str:
        return self.__repr__()

    def sel(self, *tup):
        """Select a sample by matching coordinate values."""
        from functools import reduce
        import operator

        query = reduce(
            operator.and_,
            (self.feature_ds[name] == val for name, val in tup),
        )
        idx = np.where(query)[0][0]
        return self[idx]

    @property
    def data(self) -> torch.Tensor:
        if self._data is None:
            raise ValueError("Failed to initialise data.")
        return self._data

    @property
    def labels(self) -> torch.Tensor:
        if self._labels is None:
            raise ValueError("Failed to initialise labels.")
        return self._labels

    def __len__(self):
        """Dataset length."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a transformed feature tensor and its label."""
        image = self.data[idx]
        if self.transform is not None:
            image = self.transform.fit_transform(image)

        label = self.labels[idx]

        return image, label


Categories = Literal["slope", "dx", "xi", "sigma", "r"]


def get_transform_state_dict_from_dataloader(dataloader) -> dict | None:
    """Extract the transform state dict from a dataloader if present."""
    if dataloader.dataset.dataset.transform is None:
        return None
    return dataloader.dataset.dataset.transform.state_dict()


def get_dataset(config: TrainDataSettings, transform=None) -> Dataset:
    """Create the default training dataset from config."""
    dataset = ArpsDataset(
        config.feature_path,
        config.target_path,
        transform=transform,
    )
    return dataset


def compute_mean_and_std(
    dataset: Dataset,
) -> tuple[Sequence[float], Sequence[float]]:
    """Compute per-channel mean and std for normalisation."""
    shape = dataset[:][0].shape
    n_features = shape[1]

    mean = dataset[:][0].mean(dim=0).view(n_features, -1).mean(dim=1).tolist()
    std = dataset[:][0].std(dim=0).view(n_features, -1).mean(dim=1).tolist()
    return mean, std


def get_seeded_dataloaders(
    dataset, config: TrainSettings, split_strat
) -> tuple[tuple[DataLoader, ...], str]:
    """Returns seeded dataloaders

    TODO: The implementation of the normalisation is a bit hacky.
    """

    train_dataset, _ = split_strat.split_dataset(dataset)

    mean, std = compute_mean_and_std(train_dataset)
    transform = NormaliseTransform(mean, std)
    dataset.transform = transform
    train_dataset, val_dataset = split_strat.split_dataset(dataset)

    return (
        (
            get_seeded_dataloader(train_dataset, config),
            get_seeded_dataloader(val_dataset, config),
            None,
            # get_seeded_dataloader(test_dataset, config),
        ),
        str(dataset),
    )


def get_seeded_dataloader(dataset: Dataset, config: TrainSettings) -> DataLoader:
    """Build a deterministic DataLoader with reproducible workers."""

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        pin_memory=config.pin_memory,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
    )


def seeded_datapipeline(
    config: Settings,
) -> tuple[tuple[DataLoader, ...], str]:
    """Create train/val loaders with a fixed seed and chosen split strategy."""
    from lightning import seed_everything

    seed_everything(config.train.seed, workers=True)
    dataset = get_dataset(config.train_data)

    split_strat = split_registry[config.train.split_strategy]

    if config.train.split_strategy == "random":
        split = split_strat(config.train.split)
    elif config.train.split_strategy == "coords":
        split = split_strat(
            config.train.train_indices_path, config.train.val_indices_path
        )
    else:
        raise KeyError(
            f"Split strategy {config.train.split_strategy} does not exist. Choose between 'random' and 'coords'"
        )

    return get_seeded_dataloaders(dataset, config.train, split)


def seeded_datapipeline_inspect(
    config: Settings,
) -> tuple[tuple[DataLoader, ...], str]:
    """Inspect loaders while also returning sample indices."""
    from lightning import seed_everything

    class ReturnIndex(ArpsDataset):
        def __getitem__(self, idx):
            image = self.data[idx]
            if self.transform is not None:
                image = self.transform.fit_transform(image)

            label = self.labels[idx]

            return image, label, idx

    seed_everything(config.train.seed, workers=True)
    dataset = ReturnIndex(
        config.train_data.feature_path,
        config.train_data.target_path,
        transform=None,
    )
    return get_seeded_dataloaders(dataset, config.train)
