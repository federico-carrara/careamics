"""Pydantic model representing CAREamics prediction configuration."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union
from typing_extensions import Self

from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from .validators import check_axes_validity, patch_size_ge_than_8_power_of_2


class InferenceConfig(BaseModel):
    """Configuration class for the prediction model."""

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    data_type: Literal["array", "tiff", "custom"]  # As defined in SupportedData
    """Type of input data: numpy.ndarray (array) or path (tiff or custom)."""

    tile_size: Optional[Union[list[int]]] = Field(
        default=None, min_length=2, max_length=3
    )
    """Tile size of prediction, only effective if `tile_overlap` is specified."""

    tile_overlap: Optional[Union[list[int]]] = Field(
        default=None, min_length=2, max_length=3
    )
    """Overlap between tiles, only effective if `tile_size` is specified."""

    axes: str
    """Data axes (TSCZYX) in the order of the input data."""
    
    norm_type: Literal["normalize", "standardize"] = "standardize"
    """Normalization type, either min-max normalization or standardization using mean
    and standard deviation."""

    image_means: Optional[list] = Field(None, min_length=0, max_length=32)
    """Mean values for each input channel."""

    image_stds: Optional[list] = Field(None, min_length=0, max_length=32)
    """Standard deviation values for each input channel."""
    
    image_mins: Optional[list] = Field(None, min_length=0)
    """Minimum values for each input channel."""
    
    image_maxs: Optional[list] = Field(None, min_length=0)
    """Maximum values for each input channel."""

    # TODO only default TTAs are supported for now
    tta_transforms: bool = Field(default=True)
    """Whether to apply test-time augmentation (all 90 degrees rotations and flips)."""

    # Dataloader parameters
    batch_size: int = Field(default=1, ge=1)
    """Batch size for prediction."""

    dataloader_params: Optional[dict] = Field(default=None, exclude=True)
    """Dictionary of PyTorch dataloader parameters."""

    
    @field_validator("tile_overlap")
    @classmethod
    def all_elements_non_zero_even(
        cls, tile_overlap: Optional[list[int]]
    ) -> Optional[list[int]]:
        """
        Validate tile overlap.

        Overlaps must be non-zero, positive and even.

        Parameters
        ----------
        tile_overlap : list[int] or None
            Patch size.

        Returns
        -------
        list[int] or None
            Validated tile overlap.

        Raises
        ------
        ValueError
            If the patch size is 0.
        ValueError
            If the patch size is not even.
        """
        if tile_overlap is not None:
            for dim in tile_overlap:
                if dim < 1:
                    raise ValueError(
                        f"Patch size must be non-zero positive (got {dim})."
                    )

                if dim % 2 != 0:
                    raise ValueError(f"Patch size must be even (got {dim}).")

        return tile_overlap

    @field_validator("tile_size")
    @classmethod
    def tile_min_8_power_of_2(
        cls, tile_list: Optional[list[int]]
    ) -> Optional[list[int]]:
        """
        Validate that each entry is greater or equal than 8 and a power of 2.

        Parameters
        ----------
        tile_list : list of int
            Patch size.

        Returns
        -------
        list of int
            Validated patch size.

        Raises
        ------
        ValueError
            If the patch size if smaller than 8.
        ValueError
            If the patch size is not a power of 2.
        """
        patch_size_ge_than_8_power_of_2(tile_list)

        return tile_list

    @field_validator("axes")
    @classmethod
    def axes_valid(cls, axes: str) -> str:
        """
        Validate axes.

        Axes must:
        - be a combination of 'STCZYX'
        - not contain duplicates
        - contain at least 2 contiguous axes: X and Y
        - contain at most 4 axes
        - not contain both S and T axes

        Parameters
        ----------
        axes : str
            Axes to validate.

        Returns
        -------
        str
            Validated axes.

        Raises
        ------
        ValueError
            If axes are not valid.
        """
        # Validate axes
        check_axes_validity(axes)

        return axes

    @model_validator(mode="after")
    def validate_dimensions(self: Self) -> Self:
        """
        Validate 2D/3D dimensions between axes and tile size.

        Returns
        -------
        Self
            Validated prediction model.
        """
        expected_len = 3 if "Z" in self.axes else 2

        if self.tile_size is not None and self.tile_overlap is not None:
            if len(self.tile_size) != expected_len:
                raise ValueError(
                    f"Tile size must have {expected_len} dimensions given axes "
                    f"{self.axes} (got {self.tile_size})."
                )

            if len(self.tile_overlap) != expected_len:
                raise ValueError(
                    f"Tile overlap must have {expected_len} dimensions given axes "
                    f"{self.axes} (got {self.tile_overlap})."
                )

            if any((i >= j) for i, j in zip(self.tile_overlap, self.tile_size)):
                raise ValueError("Tile overlap must be smaller than tile size.")

        return self

    @model_validator(mode="after")
    def std_only_with_mean(self: Self) -> Self:
        """
        Check that mean and std are either both None, or both specified.

        Returns
        -------
        Self
            Validated prediction model.

        Raises
        ------
        ValueError
            If std is not None and mean is None.
        """
        # check that mean and std are either both None, or both specified
        if not self.image_means and not self.image_stds:
            raise ValueError("Mean and std must be specified during inference.")

        if (self.image_means and not self.image_stds) or (
            self.image_stds and not self.image_means
        ):
            raise ValueError(
                "Mean and std must be either both None, or both specified."
            )

        elif (self.image_means is not None and self.image_stds is not None) and (
            len(self.image_means) != len(self.image_stds)
        ):
            raise ValueError(
                "Mean and std must be specified for each " "input channel."
            )

        return self

    def _update(self, **kwargs: Any) -> None:
        """
        Update multiple arguments at once.

        Parameters
        ----------
        **kwargs : Any
            Key-value pairs of arguments to update.
        """
        self.__dict__.update(kwargs)
        self.__class__.model_validate(self.__dict__)

    def set_3D(self, axes: str, tile_size: list[int], tile_overlap: list[int]) -> None:
        """
        Set 3D parameters.

        Parameters
        ----------
        axes : str
            Axes.
        tile_size : list of int
            Tile size.
        tile_overlap : list of int
            Tile overlap.
        """
        self._update(axes=axes, tile_size=tile_size, tile_overlap=tile_overlap)
        
    def set_means_and_stds(
        self,
        image_means: Optional[Union[NDArray, tuple, list]],
        image_stds: Optional[Union[NDArray, tuple, list]],
        target_means: Optional[Union[NDArray, tuple, list]] = None,
        target_stds: Optional[Union[NDArray, tuple, list]] = None,
    ) -> None:
        """
        Set mean and standard deviation of the data across channels.

        This method should be used instead setting the fields directly, as it would
        otherwise trigger a validation error.

        Parameters
        ----------
        image_means : numpy.ndarray, tuple or list
            Mean values for normalization.
        image_stds : numpy.ndarray, tuple or list
            Standard deviation values for normalization.
        target_means : numpy.ndarray, tuple or list, optional
            Target mean values for normalization, by default ().
        target_stds : numpy.ndarray, tuple or list, optional
            Target standard deviation values for normalization, by default ().
        """
        # make sure we pass a list
        if image_means is not None:
            image_means = list(image_means)
        if image_stds is not None:
            image_stds = list(image_stds)
        if target_means is not None:
            target_means = list(target_means)
        if target_stds is not None:
            target_stds = list(target_stds)

        self._update(
            image_means=image_means,
            image_stds=image_stds,
            target_means=target_means,
            target_stds=target_stds,
        )
    
    @model_validator(mode="after")
    def min_only_with_max(self: Self) -> Self:
        """
        Check that min and max are either both None, or both specified.

        Returns
        -------
        Self
            Validated data model.
        
        Raises
        ------
        ValueError
            If min is not None and max is None.
        """
        # check that min and max are either both None, or both specified
        if (self.image_mins and not self.image_maxs) or (
            self.image_maxs and not self.image_mins
        ):
            raise ValueError(
                "Min and max must be either both None, or both specified."
            )

        elif (self.image_mins is not None and self.image_maxs is not None) and (
            len(self.image_mins) != len(self.image_maxs)
        ):
            raise ValueError("Min and max must be specified for each input channel.")

        return self
            
    def set_mins_and_maxs(
        self,
        image_mins: Union[NDArray, tuple, list, None],
        image_maxs: Union[NDArray, tuple, list, None],
        target_mins: Optional[Union[NDArray, tuple, list, None]] = None,
        target_maxs: Optional[Union[NDArray, tuple, list, None]] = None,
    ) -> None:
        """
        Set min and max values of the data across channels.
        
        This method should be used instead setting the fields directly, as it would
        otherwise trigger a validation error.
        
        Parameters
        ----------
        image_mins : numpy.ndarray, tuple or list
            Minimum values for normalization.
        image_maxs : numpy.ndarray, tuple or list
            Maximum values for normalization.
        target_mins : numpy.ndarray, tuple or list, optional
            Target minimum values for normalization, by default ().
        target_maxs : numpy.ndarray, tuple or list, optional
            Target maximum values for normalization, by default ().
        """
        # make sure we pass a list
        if image_mins is not None:
            image_mins = list(image_mins)
        if image_maxs is not None:
            image_maxs = list(image_maxs)
        if target_mins is not None:
            target_mins = list(target_mins)
        if target_maxs is not None:
            target_maxs = list(target_maxs)
        
        self._update(
            image_mins=image_mins,
            image_maxs=image_maxs,
            target_mins=target_mins,
            target_maxs=target_maxs,
        )
        
    @model_validator(mode="after")
    def _validate_norm_statistics(self: Self) -> Self:
        """Validate that normalization statistics are correctly provided."""
        if self.norm_type == "standardize":
            if not self.image_means or not self.image_stds:
                raise ValueError(
                    "Mean and std must be both provided for standardization."
                )
        elif self.norm_type == "normalize":
            if not self.image_mins or not self.image_maxs:
                raise ValueError(
                    "Min and max must be both provided for normalization."
                )
