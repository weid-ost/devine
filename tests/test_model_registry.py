import tempfile
from pathlib import Path

import pytest

from devine_torch.model_configs import ModelInfo
from devine_torch.model_registry import (
    ModelRegistry,
    load_weight_json,
    registry,
)


def test_registry_initialization():
    """Test that registry initializes correctly."""
    temp_path = Path(tempfile.mkdtemp())
    test_registry = ModelRegistry(models_path=temp_path)
    assert test_registry.models_path == temp_path


def test_registry_nonexistent_path():
    """Test that registry initializes correctly."""
    with pytest.raises(FileNotFoundError, match="Models directory not found"):
        ModelRegistry(models_path=Path("nonexistent_path"))


def test_list_models():
    """Test listing available models."""
    models = registry.list_models()
    assert isinstance(models, list)
    model_names = set([model for model, _ in models])
    assert "devine-low-bias" in model_names


def test_contains_method():
    """Test __contains__ method."""
    assert "devine-low-bias" in registry
    assert "devine-low-scatter" in registry

    # In case a type checker is used, this will be caught already, see ModelInfo
    assert "invalid_model" not in registry


def test_get_model_valid():
    """Test getting a valid model."""
    model_info = registry.get_model_info("devine-low-bias")
    assert isinstance(model_info, ModelInfo)
    assert model_info.in_channels == 3


def test_get_model_invalid():
    """Test getting invalid model raises error."""
    with pytest.raises(KeyError):
        # In case a type checker is used, this will be caught already, see ModelInfo
        registry.get_model_info("nonexistent_model")


def test_load_weight_json_basic(tmp_path):
    """Test basic weight loading."""
    weights_data = {"layer1.weight": [[1.0, 2.0]], "layer1.bias": [0.1]}
    weights_file = tmp_path / "weights.json"

    import json

    with open(weights_file, "w") as f:
        json.dump(weights_data, f)

    weights = load_weight_json(weights_file)
    assert "layer1.weight" in weights
    assert "layer1.bias" in weights
