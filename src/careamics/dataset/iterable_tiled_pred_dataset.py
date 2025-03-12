"""Iterable tiled prediction dataset used to load data file by file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Generator, Optional

from numpy.typing import NDArray
from torch.utils.data import IterableDataset

from careamics.file_io.read import read_tiff
from careamics.transforms import Compose

from ..config import InferenceConfig
from ..config.tile_information import TileInformation
from ..config.transformations import NormalizeModel, StandardizeModel
from .dataset_utils import iterate_over_files
from .tiling import extract_tiles


class IterableTiledPredDataset(IterableDataset):
    """Tiled prediction dataset.

    Parameters
    ----------
    prediction_config : InferenceConfig
        Inference configuration.
    src_files : list of pathlib.Path
        List of data files.
    read_source_func : Callable, optional
        Read source function for custom types, by default read_tiff.
    **kwargs : Any
        Additional keyword arguments, unused.

    Attributes
    ----------
    data_path : str or pathlib.Path
        Path to the data, must be a directory.
    axes : str
        Description of axes in format STCZYX.
    mean : float, optional
        Expected mean of the dataset, by default None.
    std : float, optional
        Expected standard deviation of the dataset, by default None.
    patch_transform : Callable, optional
        Patch transform callable, by default None.
    """

    def __init__(
        self,
        prediction_config: InferenceConfig,
        src_files: list[Path],
        read_source_func: Callable = read_tiff,
        read_source_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        prediction_config : InferenceConfig
            Inference configuration.
        src_files : List[Path]
            List of data files.
        read_source_func : Callable, optional
            Read source function for custom types, by default read_tiff.
        read_source_kwargs : Dict[str, Any], optional
            Additional keyword arguments for the read function, by default None.
        **kwargs : Any
            Additional keyword arguments, unused.

        Raises
        ------
        ValueError
            If mean and std are not provided in the inference configuration.
        """
        if (
            prediction_config.tile_size is None
            or prediction_config.tile_overlap is None
        ):
            raise ValueError(
                "Tile size and overlap must be provided for tiled prediction."
            )

        self.prediction_config = prediction_config
        self.data_files = src_files
        self.axes = prediction_config.axes
        self.tile_size = prediction_config.tile_size
        self.tile_overlap = prediction_config.tile_overlap
        self.read_source_func = read_source_func
        self.read_source_kwargs = read_source_kwargs
        self.norm_type = self.prediction_config.norm_type
        self.image_means = self.prediction_config.image_means
        self.image_stds = self.prediction_config.image_stds
        self.image_mins = self.prediction_config.image_mins
        self.image_maxs = self.prediction_config.image_maxs

        # create normalization transform
        if self.norm_type == "normalize":
            norm_transform = NormalizeModel(
                image_mins=self.image_mins,
                image_maxs=self.image_maxs,
            )
        elif self.norm_type == "standardize":
            norm_transform = StandardizeModel(
                image_means=self.image_means,
                image_stds=self.image_stds
            )

        self.patch_transform = Compose(transform_list=[norm_transform])
        
    def __iter__(
        self,
    ) -> Generator[tuple[NDArray, TileInformation], None, None]:
        """
        Iterate over data source and yield single patch.

        Yields
        ------
        Generator of NDArray and TileInformation tuple
            Generator of single tiles.
        """
        assert (
            self.image_means is not None and self.image_stds is not None
        ), "Mean and std must be provided"

        for sample, _, sample_id in iterate_over_files(
            self.prediction_config,
            self.data_files,
            read_source_func=self.read_source_func,
            read_source_kwargs=self.read_source_kwargs,
        ):
            # generate patches, return a generator of single tiles
            patch_gen = extract_tiles(
                arr=sample,
                tile_size=self.tile_size,
                overlaps=self.tile_overlap,
                file_id=sample_id,
            )

            # apply transform to patches
            for patch_array, tile_info in patch_gen:
                transformed_patch, _ = self.patch_transform(patch=patch_array)

                yield transformed_patch, tile_info
