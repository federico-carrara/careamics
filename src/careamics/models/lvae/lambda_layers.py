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
        fluorophores: Sequence[str],
        wv_range: Sequence[int],
        ref_learnable: bool = False,
        num_bins: int = 32,
        burn_in: int = 0,
    ):
        """
        Parameters
        ----------
        fluorophores : Sequence[str]
            A sequence of fluorophore names.
        wv_range : Sequence[int]
            The wavelength range of the spectral image.
        ref_learnable : bool, optional
            Whether to make the reference matrix learnable. Default is `False`.
        num_bins : int, optional
            The number of bins to use for the reference matrix. Default is 32.
        burn_in : int, optional
            The number of burn-in steps before starting learning the reference matrix.
            Default is 0.
        """
        super().__init__()
        self.fluorophores = fluorophores
        self.wv_range = wv_range
        self.ref_learnable = ref_learnable
        self.num_bins = num_bins
        self.burn_in = burn_in
        
        # get the reference matrix from FPBase
        matrix = FPRefMatrix(
            fp_names=self.fluorophores,
            n_bins=self.num_bins,
            interval=self.wv_range
        )
        self.ref_matrix = nn.Parameter(
            matrix.create(), requires_grad=self.ref_learnable and self.burn_in == 0
        )
    
    def update_learnability(self, curr_epoch: int) -> None:
        """Update the reference matrix learnability.
        
        Parameters
        ----------
        curr_epoch : int
            The current epoch.
        """
        if self.ref_matrix.requires_grad or not self.ref_learnable:
            return
        
        if curr_epoch + 1 >= self.burn_in:
            self.ref_matrix.requires_grad_(True)
        
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
    

