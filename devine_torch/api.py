"""High-level prediction API used by the CLI."""

import warnings
from pathlib import Path

import torch
from xarray import Dataset, load_dataset

from devine_torch.data.dataset import Transform
from devine_torch.model_registry import registry
from devine_torch.transform import create_flowfield

device = "cuda" if torch.cuda.is_available() else "cpu"


def predict(model_name, netcdf_file, output_file: Path | str = "flowfield.nc"):
    """Run inference for a given config and write the predicted flowfield to disk."""

    if model_name is None:
        raise ValueError(
            f"Model name is not provided.Available models: {registry.list_models()}"
        )

    if model_name not in registry:
        raise ValueError(
            f"Model {model_name} is not a valid model name. "
            f"Available models: {registry.list_models()}"
        )

    if netcdf_file is None:
        raise FileNotFoundError("Please provide a valid netcdf_file.")

    model, transform = registry.load_model(model_name)
    ds = load_dataset(netcdf_file)

    if len(ds.data_vars) != model.hparams["input_channels"]:
        raise ValueError(
            f"""Model {model_name} expects input to have {model.hparams["input_channels"]} features, but got input with {len(ds.data_vars)} (data_vars: {list(ds.data_vars)})"""
        )

    x = get_torch_tensor(ds, transform=transform)

    model.to(device)
    model.eval()

    _to_concat = []
    batch_size = 8
    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*padding=.*")
            for batch_idx in range(0, len(x), batch_size):
                _to_concat.append(
                    model(x[batch_idx : batch_idx + batch_size].to(device)).cpu()
                )

    y_hat = torch.concat(_to_concat)

    ds_pred = create_flowfield(y_hat.cpu().numpy(), list(ds.dims), ds.coords)
    ds_pred.to_netcdf(output_file)


def get_torch_tensor(ds: Dataset, transform: Transform | None = None) -> torch.Tensor:
    """Convert an Xarray dataset to a torch tensor in NCHW order and apply an optional transform."""
    data = torch.from_numpy(
        ds.to_dataarray().transpose("n", "variable", "nrows", "ncols").values
    ).to(torch.float32)
    if transform is not None:
        return transform.fit_transform(data)
    return data
