from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import xarray as xr

from devine_torch.api import get_torch_tensor, predict


def test_get_torch_tensor_without_transform():
    """Test converting xarray Dataset to torch tensor without transform."""
    # Create simple test data
    data = np.random.rand(2, 3, 5, 5).astype(np.float32)

    # Create xarray Dataset
    data_vars = {}
    for i, var in enumerate(["var1", "var2", "var3"]):
        data_vars[var] = (("n", "nrows", "ncols"), data[:, i, :, :])

    coords = {"n": range(2), "nrows": range(5), "ncols": range(5)}

    ds = xr.Dataset(data_vars, coords=coords)

    result = get_torch_tensor(ds, transform=None)

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert result.shape == (2, 3, 5, 5)

    assert torch.allclose(result, torch.tensor(data))


def test_get_torch_tensor_with_transform():
    """Test converting xarray Dataset with transform."""
    # Create simple dataset
    data = np.ones((1, 2, 3, 3)).astype(np.float32)

    data_vars = {}
    for i, var in enumerate(["var1", "var2"]):
        data_vars[var] = (("n", "nrows", "ncols"), data[:, i, :, :])

    coords = {"n": range(1), "nrows": range(3), "ncols": range(3)}
    ds = xr.Dataset(data_vars, coords=coords)

    # Mock transform
    mock_transform = Mock()
    mock_transform.fit_transform.return_value = torch.tensor(data * 2)

    result = get_torch_tensor(ds, transform=mock_transform)

    mock_transform.fit_transform.assert_called_once()

    assert torch.allclose(result, torch.tensor(data * 2))


def test_predict_model_name_none():
    """Test predict with None model name."""
    settings = Mock()
    settings.app.model = None

    with pytest.raises(ValueError, match="Model name is not provided"):
        predict(settings.app.model, settings.app.netcdf_file)


@patch("devine_torch.model_registry.registry", autospec=True)
def test_predict_invalid_model_name(mock_registry):
    """Test predict with invalid model name."""
    mock_registry.__contains__.return_value = False
    mock_registry.list_models.return_value = ["valid_model"]

    settings = Mock()
    settings.app.model = "invalid_model"

    with pytest.raises(ValueError, match="invalid_model is not a valid model name"):
        predict(settings.app.model, settings.app.netcdf_file)


@patch("devine_torch.model_registry.registry", autospec=True)
def test_predict_no_netcdf_file(mock_registry):
    """Test predict with no NetCDF file provided."""
    settings = Mock()
    settings.app.model = "devine-low-bias"
    settings.app.netcdf_file = None

    mock_registry.__contains__.return_value = True

    with pytest.raises(FileNotFoundError, match="Please provide a valid netcdf_file"):
        predict(settings.app.model, settings.app.netcdf_file)
