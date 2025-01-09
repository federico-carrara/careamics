from typing import Optional, Sequence

import numpy as np
from numpy import ndarray as NDArray


class SyntheticNoise:
    """Add synthetic noise to the image.
    
    The noise added to the image can be Gaussian (read-out noise) and/or
    Poisson (shot noise).
    
    Given \( F \) = `poisson_noise_factor`, for a pixel of intensity \( I \), the
    resulting pixel intensity with Poisson noise will have:
    
    .. math::
        \mu = I
        \sigma = noise = \sqrt{I / F}

    Hence, the Poisson noise increase with the inverse of the square root of F.
    Namely, for F \in (0, 1), the resulting Poisson noise will be larger than in the
    original  pixel.
    
    Gaussian noise is simply drawn from a normal distribution with mean 0 and standard
    deviation equal to the `gaussian_noise_factor` and added to the pixel intensity.
    
    Attributes
    ----------
    poisson_noise_factor : Optional[float]
        A factor determining the magnitude of Poisson noise. Specifically, resulting
        Poisson noise is inversely proportional to (the square root of) this factor.
        Hence, consider using a value in (0, 1) to increase the noise. 
        If None, Poisson noise is disabled.
    gaussian_noise_factor : Optional[float]
        A multiplicative factor determining the magnitude of Gaussian noise.
        Specifically, it is used to multiply a scale value peculiar of the data
        synthetic noise is applied to (e.g., their standard deviation). 
        In this way, the noise is drawn from `N(0, gaussian_noise_factor * scale)`.
        If None, Gaussian noise is disabled.
    """
    
    def __init__(
        self,
        poisson_noise_factor: Optional[float] = None,
        gaussian_noise_factor: Optional[float] = None
    ):
        """Constructor.
        
        Parameters
        ----------
        poisson_noise_factor : Optional[float]
            A factor determining the magnitude of Poisson noise. Specifically, resulting
            Poisson noise is inversely proportional to (the square root of) this factor.
            Hence, consider using a value in (0, 1) to increase the noise. 
            If None, Poisson noise is disabled.
        gaussian_noise_factor : Optional[float]
            A multiplicative factor determining the magnitude of Gaussian noise.
            Specifically, it is used to multiply a scale value peculiar of the data
            synthetic noise is applied to (e.g., their standard deviation). 
            In this way, the noise is drawn from `N(0, gaussian_noise_factor * scale)`.
            If None, Gaussian noise is disabled.
        """        
        self.poisson_noise_factor = poisson_noise_factor
        self.gaussian_noise_factor = gaussian_noise_factor

    
    def __call__(self, arr: NDArray, axes: str) -> NDArray:
        """Apply the transform.
        
        NOTE: Poisson sampling requires the input to be positive. Hence, the method
        will raise an error if the min intensity in the input is not positive.
        
        Parameters
        ----------
        arr : NDArray
            The array to apply synthetic noise to. Shape is (S, C, Z, Y, X).
        scale : Sequence[float]
            The scale values peculiar of the input data. Specifically, the standard
            deviation of the Gaussian noise is determined by multiplying the scale
            value by the `gaussian_noise_factor`. The code assumes a scale value for
            each channel.
        axes : str
            Description of the axes in the input array (e.g., a subset of `STCZYX`).
        
        Returns
        -------
        NDArray
            The transformed array.
        """
        # --- apply Poisson noise
        if self.poisson_noise_factor:
            out_dtype = arr.dtype
            arr = np.random.poisson(arr * self.poisson_noise_factor) / self.poisson_noise_factor
            arr = arr.astype(out_dtype) # as Poisson noise is integer-valued
        
        # --- apply Gaussian noise
        if self.gaussian_noise_factor:
            # compute scale as array std
            ax_ids = [i for i, ax in enumerate(axes) if ax != "C"]
            scale = np.std(arr, axis=ax_ids, keepdims=True) * self.gaussian_noise_factor
            # add Gaussian noise
            arr += np.random.normal(0, scale, arr.shape)
        
        return arr