"""Transform helpers for rotating, cropping, and assembling flowfields."""

from typing import Hashable

import torch
from numpy.typing import NDArray
from xarray import Coordinates, DataArray, Dataset


def create_flowfield(
    flow_mat: NDArray | torch.Tensor,
    dimensions: list[str] | list[Hashable],
    coords: Coordinates,
) -> Dataset:
    """Wrap model outputs into an Xarray Dataset with u, v, w components."""

    if not flow_mat.shape[1] == 3:
        raise ValueError(
            f"The flowfield consist of 3 components, but the provided array has {flow_mat.shape[1]} components."
        )

    ucomp_arr = DataArray(data=flow_mat[:, 0], dims=dimensions, coords=coords)
    vcomp_arr = DataArray(data=flow_mat[:, 1], dims=dimensions, coords=coords)
    wcomp_arr = DataArray(data=flow_mat[:, 2], dims=dimensions, coords=coords)

    data_dict = dict(
        ucompwind=ucomp_arr,
        vcompwind=vcomp_arr,
        wcompwind=wcomp_arr,
    )
    return Dataset(data_dict)
