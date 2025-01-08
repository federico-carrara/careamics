"""In-memory tiled prediction dataset."""

from __future__ import annotations
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset

from careamics.file_io.read import read_tiff
from careamics.transforms import Compose

from ..config import InferenceConfig
from ..config.tile_information import TileInformation
from ..config.transformations import NormalizeModel
from .tiling import prepare_tiles, prepare_tiles_array


class InMemoryTiledPredDataset(Dataset):
    """Prediction dataset storing data in memory and returning tiles of each image.

    Parameters
    ----------
    prediction_config : InferenceConfig
        Prediction configuration.
    inputs : NDArray
        Input data.
    """

    def __init__(
        self,
        prediction_config: InferenceConfig,
        inputs: NDArray,
        read_source_func: Callable = read_tiff,
        read_source_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        prediction_config : InferenceConfig
            Prediction configuration.
        inputs : NDArray
            Input data.

        Raises
        ------
        ValueError
            If data_path is not a directory.
        """
        if (
            prediction_config.tile_size is None
            or prediction_config.tile_overlap is None
        ):
            raise ValueError(
                "Tile size and overlap must be provided to use the tiled prediction "
                "dataset."
            )

        self.pred_config = prediction_config
        self.inputs = inputs
        self.axes = self.pred_config.axes
        self.tile_size = prediction_config.tile_size
        self.tile_overlap = prediction_config.tile_overlap
        self.image_means = self.pred_config.image_means
        self.image_stds = self.pred_config.image_stds
        
        # read function
        self.read_source_func = read_source_func
        self.read_source_kwargs = read_source_kwargs

        # Generate patches
        # TODO: this is just unsupervised, need to add targets
        self.data = self._prepare_tiles()

        # get transforms
        self.patch_transform = Compose(
            transform_list=[
                NormalizeModel(image_means=self.image_means, image_stds=self.image_stds)
            ],
        )

    def _prepare_tiles(self) -> list[tuple[NDArray, TileInformation]]:
        """
        Iterate over data source and create an array of patches.

        Returns
        -------
        list[tuple[NDArray, TileInformation]]
            List of tiles and tile information.
        """
        if isinstance(self.inputs, np.ndarray):
            # get tiles from the input array
            patches_list = prepare_tiles_array(
                data=self.inputs,
                axes=self.axes,
                tile_size=self.tile_size,
                tile_overlap=self.tile_overlap,
            )
        else:
            # read the input data & then get tiles
            patches_list = prepare_tiles(
                fpaths=self.inputs,
                axes=self.axes,
                tile_size=self.tile_size,
                tile_overlap=self.tile_overlap,
                read_source_func=self.read_source_func,
                read_source_kwargs=self.read_source_kwargs,
            )

        if len(patches_list) == 0:
            raise ValueError("No tiles generated, ")

        return patches_list

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[NDArray, TileInformation]:
        """
        Return the patch corresponding to the provided index.

        Parameters
        ----------
        index : int
            Index of the patch to return.

        Returns
        -------
        tuple of NDArray and TileInformation
            Transformed patch.
        """
        tile_array, tile_info = self.data[index]

        # Apply transforms
        transformed_tile, _ = self.patch_transform(patch=tile_array)

        return transformed_tile, tile_info
