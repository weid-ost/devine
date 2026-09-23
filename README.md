# devine-torch

`devine-torch` predicts wind flow fields from terrain features using a PyTorch model. It ships a small CLI (`devine`) for listing bundled models and running inference on NetCDF inputs.

This is heavily based on [wind_downscaling_cnn](https://github.com/louisletoumelin/wind_downscaling_cnn) and

Le Toumelin, L., I. Gouttevin, N. Helbig, C. Galiez, M. Roux, and F. Karbou, 2023.
Emulating the Adaptation of Wind Fields to Complex Terrain with Deep Learning.
Artif. Intell. Earth Syst., 2, e220034, https://doi.org/10.1175/AIES-D-22-0034.1.

The architecture of the UNet (DEVINE) has been implemented in PyTorch.


## Table of contents
- [Requirements](#requirements)
- [Quick setup with uv](#quick-setup-with-uv)
- [Install for CLI-only use](#install-for-cli-only-use)
- [CLI usage](#cli-usage)
- [Run a prediction](#run-a-prediction)
- [DEVINE input and output](#devine-input-and-output)
  - [Input NetCDF format (required)](#input-netcdf-format-required)
  - [Input generation](#input-generation)
  - [Output NetCDF structure](#output-netcdf-structure)
  - [Output wind speed scaling](#output-wind-speed-scaling)
- [Packaged models](#packaged-models)
- [Project layout](#project-layout)

## Requirements
- Python 3.10+ and `git`.
- macOS/Linux/WSL recommended. GPU is optional; CPU works (just slower).

## Quick setup with uv
1) Install `uv` if needed: `pip install uv` (or follow https://docs.astral.sh/uv/).
2) Create and activate a virtual environment:
```
uv venv .venv
source .venv/bin/activate
```
3) Install dependencies:
```
uv sync
uv pip install -e .
```

## Install for CLI-only use
If you just want the `devine` command without a dev setup:
- From a local clone (repo root): `uv tool install .`
- From Git: `uv tool install "git+ssh://git@github.com/weid-ost/devine.git"`

Make sure your shell PATH includes uv’s tool bin (shown when `uv tool install` finishes).

## CLI usage
```
devine --help
devine models
devine models --verbose
devine gpuinfo
devine predict --help
```

## Run a prediction
1) See which models are bundled:
```
devine models
```
2) Run inference on a NetCDF file:
```
devine predict -m devine-low-bias -i path/to/input.nc
```

Notes:
- `-m/--model` defaults to `devine-low-bias`.
- The prediction output is written to `flowfield.nc` by default, otherwise to `-o/--output_file` OUTPUT.
- The number of input variables must match the model’s expected channel count.
- Currently the CLI loads the full input NetCDF into memory; there is no streaming/out-of-memory processing path yet. If you have very large inputs, plan for sufficient RAM or consider implementing chunked/streaming reads as a follow-up.

## DEVINE input and output

### Input NetCDF format (required)
- Dimensions: `n` (samples), `nrows`, `ncols` (for best results, match the grid size the model was trained on; commonly 64×64).
In case resolutions other than 64×64 are used, make sure that `nrows` and `ncols` are divisible by 8, as the UNet has 3 MaxPooling layers. Otherwise, zero padding is applied to preserve output size, which may reduce accuracy near the borders.
- Data variables: one data variable per feature; they are stacked in alphabetical order into a `variable` axis for the model.
- Example layout:
```
Dimensions:  (n: 7000, nrows: 64, ncols: 64)
Coordinates (optional):
 slope    (n)
 xi       (n)
 sigma    (n)
 r        (n)
 wind_dir (n)
Data variables:
 deflLis (n, nrows, ncols)
 reldem  (n, nrows, ncols)
 ydsc    (n, nrows, ncols)
```
- Values should be numeric arrays with no missing data. Coordinate variables are preserved and passed through to the output.

### Input generation

To generate the three features, two of them depend on the coarse wind direction and are produced using `scripts/devine_pre_processing.py`.
The script is configured to run with the example data provided in the `scripts/` folder. For use with your own data, manual adaptation of input paths and parameters is required.

### Output NetCDF structure
- Dimensions mirror the input (`n`, `nrows`, `ncols`).
- Variables:
  - `ucompwind`
  - `vcompwind`
  - `wcompwind`

### Output wind speed scaling
The predicted wind speed components are based on a coarse-scale wind speed of 3 m/s (from the ARPS training dataset).
To obtain predicted u, v, and w components for different coarse-scale wind speeds, divide the wind components by 3 and multiply them by the desired coarse-scale wind speeds.

This scaling can be performed using `scripts/devine_post_processing.py` on the DEVINE model output. 
The script is configured to run with the example data provided in the `scripts/` folder; adapt the file paths to your own data as needed.

## Packaged models
Models live under `models/<model_name>/` (for example `models/devine-low-bias/`).
Each model folder contains:
- `model.pt` — model weights (PyTorch state dict).
- `norm.json` — normalization statistics used at inference time.
- `info.yaml` — metadata used by the model registry (description, channels, activation, etc.).

### devine-low-bias (default model version)

DEVINE model with low validation bias.
Use this for aggregate statistics and metrics.

### devine-low-scatter

DEVINE model with lower absolute validation errors but larger bias than devine-low-bias.
Use this model in case per sample accuracy is important.

## Project layout
- `devine_torch/` — Python package and CLI (`devine_torch/cli.py`).
- `models/` — bundled model artifacts used by the CLI.
- `tests/` — pytest suites.
- `scripts/devine_pre_processing.py` — builds DEVINE-ready input features (ydsc, deflLis, reldem) from DEM and coarse meteo NetCDF; currently processes everything in-memory.
- `scripts/devine_post_processing.py` — scales DEVINE output (see Output wind speed scaling); currently processes everything in-memory.
