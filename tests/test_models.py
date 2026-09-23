import itertools
import warnings
from pathlib import Path

import pytest
import torch

from devine_torch.config import Settings
from devine_torch.model_registry import registry

device = "cpu"


@pytest.fixture
def test_folder():
    return Path(__file__).parent


@pytest.fixture
def settings(test_folder):
    config_toml = test_folder / "devine_test.toml"
    return Settings.new(config_toml)


def test_windmet_model_various_shapes(settings):
    model, _ = registry.load_model(settings.app.model)

    model_info = registry.get_model_info(settings.app.model)

    model.to(device)
    model.eval()

    dims = [8, 32, 64, 97, 116, 128]

    shapes = itertools.product(dims, repeat=2)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*padding=.*")
        for shape in shapes:
            x = torch.rand(1, model_info.in_channels, *shape)
            with torch.no_grad():
                y_hat = model(x)
                assert y_hat.shape[-2:] == x.shape[-2:]


def test_models():
    model_configs = registry._model_configs
    for model_name, model_info in model_configs.items():
        model, _ = registry.load_model(model_name)

        model.to(device)
        model.eval()

        x = torch.rand(1, model_info.in_channels, 64, 64)
        with torch.no_grad():
            y_hat = model(x)
            assert y_hat.shape[-2:] == x.shape[-2:]


def test_activation_error(monkeypatch):
    fail_config = registry._model_configs

    fail_config["devine-low-scatter"].activation = "aReLU"

    monkeypatch.setattr(registry, "_model_configs", fail_config)
    with pytest.raises(AttributeError):
        registry.load_model("devine-low-scatter")
