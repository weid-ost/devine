"""Model Registry

A simple registry for loading trained models in the devine_torch package.
"""

import importlib.resources
import json
from pathlib import Path

import numpy as np
import torch

from .config import ModelType
from .data.dataset import NormaliseTransform, Transform
from .model_configs import ModelInfo, get_model_configs
from .models.model import WindUNet


def load_weight_json(filename: str | Path, transpose: bool = False) -> dict:
    """Load weights from a JSON file."""
    with open(filename) as f:
        weights_ = json.load(f)
    weights = {}
    for k, v in weights_.items():
        arr = np.array(v)
        if len(arr.shape) > 1 and transpose:
            arr = np.transpose(arr, (0, 1, 3, 2))
        weights[k] = torch.from_numpy(arr)
    return weights


def get_transformer(state_dict_path: str | Path):
    """Load a normalization transformer from a JSON file."""
    with open(state_dict_path, "r") as f:
        state_dict = json.load(f)
    return NormaliseTransform(mean=state_dict["mean"], std=state_dict["std"])


class ModelRegistry:
    """Simple model registry for loading trained models."""

    def __init__(self, models_path: Path | None = None):
        self._models_path = models_path
        self._model_configs: dict[ModelType, ModelInfo] = get_model_configs(
            self.models_path
        )

    @property
    def models_path(self) -> Path:
        """Get path to models directory."""
        if self._models_path:
            if not self._models_path.exists():
                raise FileNotFoundError("Models directory not found")
            return self._models_path

        package_path = importlib.resources.files("devine_torch")
        if not isinstance(package_path, Path):
            raise ValueError(f"Package path {package_path} is not of type Path.")

        path = package_path.joinpath("build_models")

        if path.exists():
            return path

        path = package_path.parent / "models"
        if path.exists():
            return path

        raise FileNotFoundError(f"Models directory {path} not found.")

    def get_model_info(self, name: ModelType) -> ModelInfo:
        """Get model info by name."""
        if name not in self._model_configs:
            raise KeyError(
                f"Model '{name}' not found. Available: {list(self._model_configs.keys())}"
            )
        return self._model_configs[name]

    def load_model(self, name: ModelType) -> tuple[WindUNet, Transform]:
        """Load a model by name."""
        model_info = self.get_model_info(name)

        model = WindUNet(
            input_channels=model_info.in_channels, activation=model_info.activation
        )

        model_path = self.models_path / name / model_info.model_file

        if not torch.cuda.is_available():
            model.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu"))
            )
        else:
            model.load_state_dict(torch.load(model_path))
        transform = get_transformer(self.models_path / name / model_info.transform_file)
        return model, transform

    def list_models(self):
        """List all available models."""
        return [
            (
                k,
                "\n".join(
                    [
                        f"- Description: {v.description}",
                        f"- Activation: {v.activation}",
                        f"- Loss: {v.loss}",
                        f"- Status: {v.status}",
                    ]
                ),
            )
            for k, v in self._model_configs.items()
        ]

    def __contains__(self, name: ModelType) -> bool:
        return name in self._model_configs


# Global registry instance
registry = ModelRegistry()
