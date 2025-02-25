from typing import Literal, Optional, Sequence

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
        num_bins: int = 32,
        ref_learnable: bool = False,
        num_frozen_epochs: int = 0,
        add_background: Optional[Literal["random", "constant", "from_image"]] = None,
        bg_learnable: bool = False,
        bg_kwargs: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        fluorophores : Sequence[str]
            A sequence of fluorophore names.
        wv_range : Sequence[int]
            The wavelength range of the spectral image.
        num_bins : int, optional
            The number of bins to use for the reference matrix. Default is 32.
        ref_learnable : bool, optional
            Whether to make the reference matrix learnable. Default is `False`.
        num_frozen_epochs : int, optional
            The number of epochs before starting learning the reference matrix.
            Default is 0.
        add_background : Literal["random", "constant", "from_image"], optional
            Whether and how to add a background spectrum to the reference matrix.
            Specifically, "random" adds a background spectrum drawn from a uniform
            distribution, "constant" adds a constant background spectrum, and
            "from_image" adds a background spectrum extracted from an image. For more
            information, see `FPRefMatrix.add_background_spectrum`. Default is None.
        bg_learnable : bool, optional
            Whether to make the background spectrum learnable. Default is `False`.
        bg_kwargs : dict, optional
            Additional keyword arguments for the background spectrum. Default is None.
        """
        super().__init__()
        self.fluorophores = fluorophores
        self.wv_range = wv_range
        self.num_bins = num_bins
        self.ref_learnable = ref_learnable
        self.num_frozen_epochs = num_frozen_epochs
        self.add_background = add_background
        self.bg_learnable = bg_learnable
        self.bg_kwargs = bg_kwargs
        
        # get the reference matrix from FPBase
        matrix = FPRefMatrix(
            fp_names=self.fluorophores,
            n_bins=self.num_bins,
            interval=self.wv_range
        )
        ref_matrix = matrix.create()
        self.ref_matrix = nn.Parameter(
            ref_matrix,
            requires_grad=self.ref_learnable and self.num_frozen_epochs == 0
        )
        
        # add background spectrum
        if self.add_background is not None:
            ref_matrix = matrix.add_background_spectrum(
                self.add_background, **self.bg_kwargs
            )
            self.bg_spectrum = nn.Parameter(
                ref_matrix[:, -1], requires_grad=self.bg_learnable
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
        
        if curr_epoch + 1 >= self.num_frozen_epochs:
            print("\nSetting spectra reference matrix to learnable.")
            self.ref_matrix.requires_grad_(True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            The unmixed images. Shape is (B, F + 1, [Z], Y, X), where F is the number of
            fluorophores to unmix, and the last channel is the background channel.
        
        Returns
        -------
        torch.Tensor
            The mixed spectral image. Shape is (B, W, [Z], Y, X), where W is the number
            of spectral channels.
        """
        B, F, *spatial = x.shape
        if self.add_background is not None:
            F -= 1

        spectral_img_fp = torch.matmul(
            self.ref_matrix, x[:, :F, ...].view(B, F, -1)
        ).view(B, -1, *spatial)
        
        spectral_img_bg = 0
        if self.add_background is not None:
            spectral_img_bg = (
                self.bg_spectrum[None, :, None] * x[:, -1].view(B, 1, -1)
            ).view(B, -1, *spatial)
        
        return spectral_img_fp + spectral_img_bg
    

