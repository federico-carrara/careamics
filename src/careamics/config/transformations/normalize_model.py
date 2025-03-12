"""Pydantic model for the Normalize transform."""

from typing import Literal, Optional

from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

from .transform_model import TransformModel


class NormalizeModel(TransformModel):
    """
    Pydantic model used to represent Normalize transformation.

    The Normalize transform is min-max normalization of the input data.

    Attributes
    ----------
    name : Literal["Normalize"]
        Name of the transformation.
    strategy : Literal["channel-wise", "global"]
        Normalization strategy. Default is "channel-wise".

    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    name: Literal["Normalize"] = "Normalize"
    
    strategy: Literal["channel-wise", "global"] = "channel-wise"
    """Normalization strategy. Default is "channel-wise"."""
    
    image_mins: Optional[list[float]] = Field(default=None, min_length=0)
    """Minimum values of the data across channels, used for normalization."""
    
    image_maxs: Optional[list[float]] = Field(default=None, min_length=0)
    """Maximum values of the data across channels, used for normalization."""
    
    target_mins: Optional[list[float]] = Field(default=None, min_length=0)
    """Minimum values of the target data across channels, used for normalization."""
    
    target_maxs: Optional[list[float]] = Field(default=None, min_length=0)
    """Maximum values of the target data across channels, used for normalization."""

    @model_validator(mode="after")
    def validate_mins_maxs(self: Self) -> Self:
        """Validate that the mins and maxs have the same length.

        Returns
        -------
        Self
            The instance of the model.
        """
        if len(self.image_mins) != len(self.image_maxs):
            raise ValueError("The number of image mins and maxs must be the same.")

        if (self.target_mins is None) != (self.target_maxs is None):
            raise ValueError(
                "Both target mins and maxs must be provided together, or bot None."
            )

        if self.target_maxs is not None and self.target_mins is not None:
            if len(self.target_maxs) != len(self.target_mins):
                raise ValueError(
                    "The number of target mins and maxs must be the same."
                )

        return self
