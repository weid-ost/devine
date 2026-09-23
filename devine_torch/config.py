"""Configuration models for CLI, training, and project settings."""

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import DirectoryPath, Field, FilePath, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)
from typing_extensions import Annotated

from devine_torch.model_configs import ModelType


class WindmetBaseSettings(BaseSettings):
    """Base settings that load from env vars and a TOML file."""

    _toml_file: ClassVar[Path | str | None] = "windmet_config.toml"
    model_config = SettingsConfigDict(
        env_prefix="windmet_",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            env_settings,
            init_settings,
            TomlConfigSettingsSource(settings_cls, toml_file=cls._toml_file),
        )


class ProjectSettings(WindmetBaseSettings):
    """High-level metadata for naming and grouping runs."""

    name: str = "default"
    environment: Literal["dev", "prod"] = "dev"
    run_name: str | None = None


class CLISettings(WindmetBaseSettings):
    """Runtime options for prediction via the CLI."""

    model: ModelType | None = "devine"
    output_dir: DirectoryPath = Path().cwd()
    netcdf_file: FilePath | None = None
    output_name: str = "flowfield"

    @field_validator("netcdf_file")
    @classmethod
    def check_is_nc_file(cls, file: FilePath | None) -> FilePath | None:
        if file is not None and file.suffix != ".nc":
            raise ValueError(
                f"Needs to be a NetCDF file with suffix '.nc', but found file with ending '{file.suffix}'."
            )

        return file


class TrainDataSettings(WindmetBaseSettings):
    """Paths to training feature and target NetCDF files."""

    feature_path: FilePath | None = None
    target_path: FilePath | None = None

    @field_validator("feature_path", "target_path")
    @classmethod
    def check_is_nc_file(cls, file: FilePath | None) -> FilePath | None:
        if file is not None and file.suffix != ".nc":
            raise ValueError(
                f"Needs to be a NetCDF file with suffix '.nc', but found file with ending '{file.suffix}'."
            )

        return file


class TrainSettings(WindmetBaseSettings):
    """Hyperparameters and trainer options."""

    lr: float = 3e-4
    weight_decay: float = 0.0
    batch_size: int = 64
    epochs: int = 1
    split: Annotated[
        list[float],
        Field(strict=True, min_length=3, max_length=3),
    ] = [0.6, 0.2, 0.2]
    log_every_n_steps: int = 50
    ckpt_path: str | FilePath | Literal["last", "hpc"] | None = None
    accelerator: Literal["cpu", "gpu"] = "gpu"
    num_workers: int = 8
    seed: int = 20241108
    pin_memory: bool = True
    shuffle: bool = True
    activation: str = "ReLU"

    tracking_uri: str | None = "file:./mlruns"

    train_indices_path: FilePath | None = None
    val_indices_path: FilePath | None = None

    split_strategy: Literal["random", "coords"] = "random"

    @field_validator("split")
    @classmethod
    def check_tuple_bounds(cls, tup: tuple):
        for i, t in enumerate(tup):
            if t < 0.0 or t > 1.0:
                raise ValueError(
                    f"The value of items should be greater than 0.0 or less then 1.0. Value of {t} at position {i} found."
                )
        return tup

    @field_validator("split")
    @classmethod
    def check_tuple_sum(cls, tup: tuple):
        if abs(sum(tup) - 1.0) > 1e-9:
            raise ValueError(
                f"The sum of all items is {sum(tup)}, but should be equal to 1.0."
            )
        return tup

    @field_validator("train_indices_path", "val_indices_path")
    @classmethod
    def check_is_nc_file(cls, file: FilePath | None) -> FilePath | None:
        if file is not None and file.suffix != ".nc":
            raise ValueError(
                f"Needs to be a NetCDF file with suffix '.nc', but found file with ending '{file.suffix}'."
            )

        return file


class Settings(WindmetBaseSettings):
    """Root configuration grouping project, data, training, and CLI sections."""

    project: ProjectSettings = Field(
        default_factory=ProjectSettings, description="Project settings"
    )

    train_data: TrainDataSettings = Field(
        default_factory=TrainDataSettings, description="Train data settings"
    )

    train: TrainSettings = Field(
        default_factory=TrainSettings, description="Train settings"
    )

    app: CLISettings = Field(default_factory=CLISettings, description="CLI settings")

    @classmethod
    def new(cls, toml_file: Path | str) -> "Settings":
        """Load settings from a specific TOML file without changing defaults."""

        custom_cls = type(cls.__name__, (cls,), {"_toml_file": toml_file})
        return custom_cls()
