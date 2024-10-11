"""Utility functions for working with spectral data."""

from typing import Sequence
from functools import cached_property
import warnings

from pydantic import BaseModel, field_validator, model_validator
import numpy as np
import torch

from .fpbase import get_fp_emission_spectrum


class Spectrum(BaseModel):
    """A Spectrum object defined by its intensity and wavelengths.
    
    Adapted from https://github.com/tlambert03/microsim.
    """
    wavelength: torch.Tensor
    """The set of wavelength of the spectrum."""
    intensity: torch.Tensor
    """The intensity of the spectrum."""
    
    @field_validator("intensity", mode="after")
    @classmethod
    def _validate_intensity(cls, value: torch.Tensor) -> torch.Tensor:
        if not np.all(value >= 0):
            warnings.warn(
                "Clipping negative intensity values in spectrum to 0", stacklevel=2
            )
            value = np.clip(value, 0, None)
        return value

    @model_validator(mode="after")
    def _validate_shapes(cls, value: 'Spectrum') -> 'Spectrum':
        if "wavelength" in value and "intensity" in value:
            if not len(value["wavelength"]) == len(value["intensity"]):
                raise ValueError(
                    "Wavelength and intensity must have the same length"
                )
        return value
    
    def _align(self, other: 'Spectrum') -> tuple['Spectrum', 'Spectrum']:
        """Align self and other over their wavelength attributes.
        
        New wavelengths are added where necessary, with associated intensities set to 0.
        
        Parameters
        ----------
        other : Spectrum
            The Spectrum to align with.
        
        Returns
        -------
        tuple[Spectrum, Spectrum]
            The aligned self and other Spectrums.
        """
        # get union of wavelength range
        w_min = min(self.wavelength.min(), other.wavelength.min())
        w_max = max(self.wavelength.max(), other.wavelength.max())
        new_range = torch.arange(w_min, w_max + 1)

        # create interpolated intensity for self and other
        aligned_self_intensity = self._interpolate_intensity(new_range)
        aligned_other_intensity = other._interpolate_intensity(new_range)

        # return new aligned Spectrums
        aligned_self = Spectrum(wavelength=new_range, intensity=aligned_self_intensity)
        aligned_other = Spectrum(wavelength=new_range, intensity=aligned_other_intensity)

        return aligned_self, aligned_other

    def _interpolate_intensity(self, new_wavelengths: torch.Tensor) -> torch.Tensor:
        """Interpolate the intensity values over the new set of wavelengths.
        
        Missing intensity values are filled with 0.
        
        Parameters
        ----------
        new_wavelengths : torch.Tensor
            The new set of wavelengths.
        
        Returns
        -------
        torch.Tensor
            The interpolated intensity values, with 0's to fill.
        """
        # find where the new wavelengths match the existing ones
        indices = (
            new_wavelengths >= self.wavelength.min() * new_wavelengths <= self.wavelength.max()
        )
        
        # get interpolated intesities
        new_intensity = torch.zeros_like(new_wavelengths)
        new_intensity[indices] = self.intensity
        
        return new_intensity
    
    def _get_bins(self, num_bins: int, interval: Sequence[int, int],) -> torch.Tensor:
        """Get bin delimiters for the given interval.
        
        Parameters
        ----------
        num_bins : int
            The number of bins to create.
        interval : Sequence[int, int]
            The interval to create the bins for.
        
        Returns
        -------
        torch.Tensor
            The bin delimiters.
        """
        range_ = interval[1] - interval[0]
        min_bin_length = range_ // num_bins
        remainder = range_ % num_bins
        bins = [interval[0]]
        for i in range(num_bins):
            # add extra wavelengths at the beginning
            curr_bin_length = min_bin_length if i >= remainder else min_bin_length + 1
            bins.append(bins[-1] + curr_bin_length)
        return torch.tensor(bins)
    
    def bin_intensity(self, num_bins: int) -> torch.Tensor:
        """Bins the intensity values according to the provided bins for the wavelength.
        
        Parameters
        ----------
        spec : Spectrum
            The spectrum to bin.

        Returns
        -------
        torch.Tensor
            Binned intensity values.
        """
        # initialize
        bin_edges = self._get_bins(
            num_bins=num_bins,
            interval=(self.wavelength.min(), self.wavelength.max())
        )
        binned_intensity = torch.zeros(len(bin_edges) - 1)

        # digitize the wavelength tensor into bin indices
        bin_indices = torch.bucketize(self.wavelength, bin_edges, right=False)

        # perform the binning
        for i in range(1, len(bin_edges)):
            mask = bin_indices == i
            binned_intensity[i - 1] = self.intensity[mask].sum()
        
        return binned_intensity
    
    
class FPRefMatrix(BaseModel):
    """
    This class is used to create a reference matrix of fluorophore emission spectra
    for the spectral unmixing task.
    """
    fp_names: Sequence[str]
    """The names of the fluorophores to include in the reference matrix."""
    n_bins: int = 32
    """The number of wavelength bins to use for the FP spectra."""
    
    @cached_property
    def fp_spectra(self) -> list[Spectrum]:
        """Aligned fluorophore emission spectra."""
        # fetch emission spectra
        fp_sp = [get_fp_emission_spectrum(fp_name) for fp_name in self.fp_names]
        
        # align spectra
        fp_sp0, *rest = fp_sp
        align_fp_sp = []
        for sp in rest:
            fp_sp0, sp = fp_sp0._align(sp)
            align_fp_sp.append(sp)
        align_fp_sp = [fp_sp0] + align_fp_sp
        
        return align_fp_sp
    
    @cached_property
    def binned_fp_intensities(self) -> list[torch.Tensor]:
        """Binned fluorophore emission spectrum intensities."""
        return [
            fp_spectrum.bin_intensity(self.n_bins) for fp_spectrum in self.fp_spectra
        ]
    
    def _normalize(self) -> list[torch.Tensor]:
        """Normalize the binned intensities of the emission spectra."""
        return [
            (curr - curr.min()) / (curr.max() - curr.min())
            for curr in self.binned_fp_intensities
        ]
        
    def create(self) -> torch.Tensor:
        """Create the reference matrix.
        
        The shape of the matrix is [W, F], where W is the number of wavelength bins
        and F is the number of fluorophores.
        """
        normalized_fp_intensities = self._normalize()
        return torch.stack(
            [intensity for intensity in normalized_fp_intensities], 
            axis=0
        )