from typing import Sequence, TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    import torch

class SpectralMixer(nn.Module):
    """
    Spectral Mixer to recombine the unmixed images into the mixed spectral image.
    
    Attributes
    ----------
    ref_matrix : nn.Parameter
        The reference matrix to use for the spectral mixing. Shape is (W, F), where W
        is the number of spectral bandss and F is the number of fluorophores to unmix.
    """
    def __init__(
        self,
        flurophores: Sequence[str],
        ref_learnable: bool = False,
    ):
        # get the reference matrix from FPBase
        self.ref_matrix = nn.Parameter(...)
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            The unmixed images. Shape is (B, F, Y, X), where F is the number of
            fluorophores to unmix.
        
        Returns
        -------
        torch.Tensor
            The mixed spectral image. Shape is (B, W, Y, X), where W is the number of
            spectral channels.
        """
        b, _, h, w = x.shape[-2:] 
        return torch.matmul(self.ref_matrix, x.view(b, p, -1)).view(b, n, h, w)
    

