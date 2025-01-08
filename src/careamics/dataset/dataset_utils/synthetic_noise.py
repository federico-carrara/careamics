from typing import Optional

import numpy as np
from numpy import ndarray as NDArray


class SyntheticNoise:
    """Add synthetic noise to the image.
    
    The noise added to the image can be Gaussian (read-out noise) and/or
    Poisson (shot noise).
    
    NOTE: Given \( F \) = `poisson_noise_factor`, for a pixel of intensity \( I \), the
    resulting pixel intensity with Poisson noise will have:
    
    .. math::
        \mu = I
        \sigma = noise = \sqrt{I / F}

    Hence, the Poisson noise increase with the inverse of the square root of F.
    Namely, for F \in (0, 1), the resulting Poisson noise will be larger than in the original
    pixel.
    
    
    Parameters
    ----------
    poisson_noise_factor : Optional[float]
        A multiplicative factor for the Poisson noise that determines the noise level.
        Specifically, Poisson noise is inversely proportional to the factor. Hence,
        consider using a value in (0, 1) to increase the noise. 
        If None, Poisson noise is disabled.
    gaussian_scale : float
        A multiplicative factor for the Gaussian noise. A sensible value is the standard
        deviation of the input data or multiples of it.
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
            A multiplicative factor for the Poisson noise that determines the noise level.
            Specifically, Poisson noise is inversely proportional to the factor. Hence,
            consider using a value in (0, 1) to increase the noise. 
            If None, Poisson noise is disabled.
        gaussian_scale : float
            A multiplicative factor for the Gaussian noise. A sensible value is the standard
            deviation of the input data or multiples of it.
            If None, Gaussian noise is disabled.
        """        
        self.poisson_noise_factor = poisson_noise_factor
        self.gaussian_noise_factor = gaussian_noise_factor
        
    
    def __call__(
        self,
        inp_arr: NDArray,
        tar_arr: Optional[NDArray] = None,
    ) -> tuple[NDArray, Optional[NDArray]]:
        """Apply the transform.
        
        Parameters
        ----------
        inp_arr : NDArray
            The input array to apply synthetic noise to.
        tar_arr : Optional[NDArray], optional
            The target array to apply synthetic noise to.
        
        Returns
        -------
        tuple[NDArray, Optional[NDArray]]
            The transformed input and target (if provided) arrays.
        """
        inp_arr = self._apply(inp_arr)
        
        if tar_arr is not None:
            tar_arr = self._apply(tar_arr)
        
        return inp_arr, tar_arr
    
    
    def _apply(self, arr: NDArray) -> NDArray:
        """Apply the transform.
        
        NOTE: Poisson sampling requires the input to be positive. Hence, the method
        will raise an error if the min intensity in the input is not positive.
        
        Parameters
        ----------
        arr : NDArray
            The array to apply synthetic noise to.
        
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
            arr += np.random.normal(0, self.gaussian_noise_factor, arr.shape)
        
        return arr