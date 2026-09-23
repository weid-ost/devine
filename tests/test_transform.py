import numpy as np
import pytest
import torch
import xarray as xr

from devine_torch.transform import (
    create_flowfield,
)


@pytest.fixture
def dummy_dataset():
    n, nrows, ncols = 3, 64, 64
    data = np.random.rand(n, nrows, ncols).astype(np.float32)
    da = xr.DataArray(data, dims=["n", "nrows", "ncols"])
    return da.to_dataset(name="dummy")


@pytest.fixture
def dummy_tensor():
    return torch.rand(1, 3, 64, 64)


def test_create_flowfield(dummy_tensor, dummy_dataset):
    coords = dummy_dataset.coords
    dims = list(dummy_dataset.dims)
    result = create_flowfield(dummy_tensor.numpy(), dims, coords)
    assert "ucompwind" in result.data_vars
    assert "vcompwind" in result.data_vars
    assert "wcompwind" in result.data_vars
