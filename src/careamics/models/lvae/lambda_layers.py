from typing import Sequence

import torch
import torch.nn as nn

from careamics.utils.spectral import FPRefMatrix


class SpectralMixer(nn.Module):
    """
    Spectral Mixer to recombine the unmixed images into the mixed spectral image.
    
    Attributes
    ----------
    ref_matrix : nn.Parameter
        The reference matrix to use for the spectral mixing. Shape is (W, F), where W
        is the number of spectral bands and F is the number of fluorophores to unmix.
    """
    def __init__(
        self,
        flurophores: Sequence[str],
        wv_range: Sequence[int],
        ref_learnable: bool = False,
        num_bins: int = 32,
    ):
        """
        Parameters
        ----------
        flurophores : Sequence[str]
            A sequence of fluorophore names.
        wv_range : Sequence[int]
            The wavelength range of the spectral image.
        ref_learnable : bool, optional
            Whether to make the reference matrix learnable. Default is `False`.
        num_bins : int, optional
            The number of bins to use for the reference matrix. Default is 32.
        """
        super().__init__()
        
        # get the reference matrix from FPBase
        matrix = FPRefMatrix(fp_names=flurophores, n_bins=num_bins, interval=wv_range)
        self.ref_matrix = nn.Parameter(matrix.create(), requires_grad=ref_learnable)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            The unmixed images. Shape is (B, F, [Z], Y, X), where F is the number of
            fluorophores to unmix.
        
        Returns
        -------
        torch.Tensor
            The mixed spectral image. Shape is (B, W, [Z], Y, X), where W is the number
            of spectral channels.
        """
        B, F, *spatial = x.shape 
        return torch.matmul(self.ref_matrix, x.view(B, F, -1)).view(B, -1, *spatial)
    

