from typing import Literal, Optional    
    
from pydantic import ConfigDict, Field    

from .transform_model import TransformModel


class SyntheticNoiseModel(TransformModel):
    """Pydantic model to represent adding synthetic noise to the image.
    
    The noise added to the image can be Gaussian (read-out noise) and/or
    Poisson (shot noise).
    
    NOTE: Given \( F \) = `poisson_noise_factor`, for a pixel of intensity \( I \), the
    resulting pixel intensity with Poisson noise will have:
    
    .. math::
        \mu = I
        \sigma = noise = \sqrt{I / F}

    Hence, the Poisson noise increase with the inverse of the square root of F.
    Namely, for F < 1, the resulting Poisson noise will be larger than in the original
    pixel.
    
    
    Attributes
    ----------
    poisson_noise_factor : Optional[float]
        A multiplicative factor for the Poisson noise that determines the noise level.
        Specifically, Poisson noise is inversely proportional to the factor. Hence,
        consider using a value < 1 to increase the noise. 
        If None, Poisson noise is disabled.
    gaussian_scale : float
        A multiplicative factor for the Gaussian noise. Assuming the input being
        standardized, the resulting noise is drawn from a Gaussian distribution
        N(0, `gaussian_noise_factor`). If None, Gaussian noise is disabled.
    """
    
    model_config = ConfigDict(
        validate_assignment=True
    )
    
    name: Literal["SyntheticNoise"] = "SyntheticNoise"
    
    poisson_noise_factor: Optional[float] = Field(None, gt=0)
    """A multiplicative factor for the Poisson noise that determines the noise level.
    Specifically, Poisson noise is inversely proportional to the factor. Hence,
    consider using a value < 1 to increase the noise. 
    If None, Poisson noise is disabled."""
    
    gaussian_noise_factor: Optional[float] = Field(default=None, gt=0)
    """A multiplicative factor for the Gaussian noise. Assuming the input being
    standardized, the resulting noise is drawn from a Gaussian distribution
    N(0, `gaussian_noise_factor`)."""
    
    