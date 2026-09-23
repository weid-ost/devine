"""Model Configurations

Configuration file that defines all available models for the registry.
"""

from pathlib import Path
from typing import Literal, TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    ModelType = Literal["devine-low-bias", "devine-low-scatter"]
else:
    ModelType = Any

class ModelInfo(BaseModel):
    """Metadata describing a packaged model and its artifacts."""

    model_file: Path = Field(alias="checkpoint")
    transform_file: Path = Field(alias="normalisation")

    in_channels: int = Field(alias="channels")
    activation: str = "ReLU"
    loss: str | None = None

    id: str | None = None
    description: str | None = None

    status: Literal["TBD", "archive", "dev", "production"] | None = None


def get_model_configs(models_path: Path):
    import yaml

    models = {}
    for m_path in models_path.iterdir():
        with open(m_path / "info.yaml", "r") as f:
            info = yaml.safe_load(f)

        models[m_path.name] = ModelInfo.model_validate(info)

    return models
