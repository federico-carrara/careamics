"""In-memory prediction dataset."""

from __future__ import annotations
from typing import Any, Callable, Optional

from numpy.typing import NDArray
from torch.utils.data import Dataset

from careamics.file_io.read import read_tiff
from careamics.transforms import Compose

from ..config import InferenceConfig
from ..config.transformations import NormalizeModel, StandardizeModel
from .dataset_utils import reshape_array


class InMemoryPredDataset(Dataset):
    """Simple prediction dataset returning images along the sample axis.

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
        self.pred_config = prediction_config
        self.input_array = inputs
        self.axes = self.pred_config.axes
        self.norm_type = self.pred_config.norm_type
        self.image_means = self.pred_config.image_means
        self.image_stds = self.pred_config.image_stds
        self.image_mins = self.pred_config.image_mins
        self.image_maxs = self.pred_config.image_maxs
        
        # read function
        self.read_source_func = read_source_func
        self.read_source_kwargs = read_source_kwargs

        # Reshape data
        self.data = reshape_array(self.input_array, self.axes)

        # get transforms
        if self.norm_type == "normalize":
            norm_transform = NormalizeModel(
                image_mins=self.image_mins,
                image_maxs=self.image_maxs,
            )
        elif self.norm_type == "standardize":
            norm_transform = StandardizeModel(
                image_means=self.image_means,
                image_stds=self.image_stds,
            )
        
        self.patch_transform = Compose(transform_list=[norm_transform],)

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> NDArray:
        """
        Return the patch corresponding to the provided index.

        Parameters
        ----------
        index : int
            Index of the patch to return.

        Returns
        -------
        NDArray
            Transformed patch.
        """
        transformed_patch, _ = self.patch_transform(patch=self.data[index])

        return transformed_patch
