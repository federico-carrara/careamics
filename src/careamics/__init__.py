"""Main CAREamics module."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("careamics")
except PackageNotFoundError:
    __version__ = "uninstalled"

__all__ = [
    "CAREamist",
    "CAREamicsKiln",
    "CAREamicsModule",
    "Configuration",
    "load_configuration",
    "save_configuration",
]

from .careamist import CAREamist
from .config import Configuration, load_configuration, save_configuration
from .lightning_datamodule import (
    CAREamicsPredictDataModule,
    CAREamicsTrainDataModule,
)
from .lightning_module import CAREamicsModule
from .lightning_prediction import CAREamicsFiring
