"""Application layer

Module containing the application logic for the model training and the inference tasks.
"""

from pathlib import Path

import click
from loguru import logger

from devine_torch.__about__ import version
from devine_torch.api import predict as devine_predict


@click.group(
    help=f"""devine {version}

    Devine predicts near-surface wind fields using deep learning.
    It is a statistical downscaling model trained on an ensemble of atmospheric wind model simulations over synthetic topographies."""
)
@click.version_option()
def cli():
    """Devine predicts near-surface wind fields using deep learning.
    It is a statistical downscaling model trained on an ensemble of
    atmospheric wind model simulations over synthetic topographies.
    """
    pass


@cli.command()
@click.option(
    "-m",
    "--model",
    type=str,
    default="devine-low-bias",
    help="Model for prediction (devine-low-bias, devine-low-scatter)",
)
@click.option(
    "-i",
    "--input_file",
    type=click.Path(exists=True),
    default=None,
    help="Input NetCDF file",
)
@click.option(
    "-o",
    "--output_file",
    type=click.Path(),
    default=Path("flowfield.nc"),
    help="Output NetCDF file",
)
def predict(model, input_file, output_file):
    """
    Prediction function that takes a NetCDF file and
    outputs a NetCDF file with the u, v and w wind speed components.

    """

    logger.info("Predicting wind speed components...")

    devine_predict(model, input_file, output_file)

    logger.info(f"Done. Output written to file {output_file}")


@cli.command()
def gpuinfo():
    """Print a quick summary of available GPUs."""
    import torch

    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")


@cli.command()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose mode")
def models(verbose):
    """List bundled models and optional metadata."""
    from devine_torch.model_registry import registry

    print("Available models:\n")

    for i, (model, info) in enumerate(registry.list_models()):
        print(f" {i + 1}) {model}")
        if verbose:
            print(f"{info}")
            print()
