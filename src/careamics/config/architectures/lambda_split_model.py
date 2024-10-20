"""λSplit Pydantic model."""

from .lvae_model import LVAEModel

from pydantic import Field

class LambdaSplitConfig(LVAEModel):
    """λSplit model.
    
    Based on `LVAEModel` class, it adds some attributes for λSplit.
    """
    
    fluorophores: list[str]
    """A list of the fluorophore names in the image to unmix."""
    ref_learnable: bool = Field(default=False)
    """Whether the reference spectra matrix is learnable."""
    num_bins: int = Field(default=32)
    """Number of bins for the spectral data."""
    