"""λSplit Pydantic model."""

from typing import Literal

from pydantic import ConfigDict, Field

from .lvae_model import LVAEModel


class LambdaSplitConfig(LVAEModel):
    """λSplit model.
    
    Based on `LVAEModel` class, it adds some attributes for λSplit.
    """
    
    model_config = ConfigDict(validate_assignment=True, validate_default=True)
    
    architecture: Literal["lambdasplit"] = Field(default="lambdasplit")
    
    fluorophores: list[str]
    """A list of the fluorophore names in the image to unmix."""
    
    ref_learnable: bool = Field(default=False)
    """Whether the reference spectra matrix is learnable."""
    
    num_bins: int = Field(default=32)
    """Number of bins for the spectral data."""
    