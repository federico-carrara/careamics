"""Utility functions for working with spectral data."""

import warnings
from functools import cached_property
from typing import Optional, Sequence

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .fpbase import get_fp_emission_spectrum


class Spectrum(BaseModel):
    """A Spectrum object defined by its intensity and wavelengths.
    
    Adapted from https://github.com/tlambert03/microsim.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    wavelength: torch.Tensor
    """The set of wavelength of the spectrum."""
    intensity: torch.Tensor
    """The intensity of the spectrum."""

    @field_validator("intensity", mode="after")
    @classmethod
    def _validate_intensity(cls, value: torch.Tensor) -> torch.Tensor:
        if not torch.all(value >= 0):
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

    @classmethod
    def from_fpbase(cls, name: str) -> "Spectrum":
        data = get_fp_emission_spectrum(name)
        return cls(wavelength=data[:, 0], intensity=data[:, 1])

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
        indices = (new_wavelengths >= self.wavelength.min()) * (new_wavelengths <= self.wavelength.max())

        # get interpolated intesities
        new_intensity = torch.zeros_like(new_wavelengths)
        new_intensity[indices] = self.intensity

        return new_intensity

    def _get_bins(self, num_bins: int, interval: Sequence[int],) -> torch.Tensor:
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
            # add extra wavelengths (reminder) at the beginning
            curr_bin_length = min_bin_length if i >= remainder else min_bin_length + 1
            bins.append(bins[-1] + curr_bin_length)
        return torch.tensor(bins)

    def bin_intensity(
        self, num_bins: int, interval: Optional[Sequence[int]] = None
    ) -> torch.Tensor:
        """Bins the intensity values according to the provided bins for the wavelength.
        
        Parameters
        ----------
        num_bins: int
            The number of bins to use.
        interval: Optional[Sequence[int]]
            Interval of wavelengths in which binning is done. Wavelengths outside this
            interval are ignored. If `None`, the interval is set to the range of the
            wavelength. Default is `None`.

        Returns
        -------
        torch.Tensor
            Binned intensity values.
        """
        if not interval:
            interval = (self.wavelength.min(), self.wavelength.max())
        
        # initialize
        bin_edges = self._get_bins(
            num_bins=num_bins,
            interval=interval,
        )
        binned_intensity = torch.zeros(len(bin_edges) - 1)

        # digitize the wavelength tensor into bin indices
        bin_indices = torch.bucketize(self.wavelength.contiguous(), bin_edges, right=False)

        # perform the binning
        for i in range(1, len(bin_edges)):
            mask = bin_indices == i
            binned_intensity[i - 1] = self.intensity[mask].sum()
            
        return binned_intensity


class FPRefMatrix(BaseModel):
    """
    This class is used to create a reference matrix of fluorophore emission spectra
    for the spectral unmixing task.
    
    Example
    -------
    # Initialize the reference matrix
    >>> fp_names = ["mCherry", "mTurquoise2", "mVenus"]
    >>> ref_matrix = FPRefMatrix(fp_names=fp_names, n_bins=32, interval=(400, 700))
    
    # Create the reference matrix
    >>> ref_matrix.create()
    """

    fp_names: Sequence[str]
    """The names of the fluorophores to include in the reference matrix."""
    n_bins: int = 32
    """The number of wavelength bins to use for the FP spectra."""
    interval: Optional[Sequence[int]] = None
    """The interval of wavelengths in which binning is done. Wavelengths outside this
    interval are ignored. If `None`, the interval is set to the range of the wavelength."""
    
    def _sort_fp_spectra(self, fp_spectra: list[Spectrum]) -> list[Spectrum]:
        """Sort the fluorophore emission spectra by wavelength."""
        def get_wavelength_at_peak_intensity(sp: Spectrum) -> float:
            return sp.wavelength[sp.intensity.argmax()]
        return sorted(fp_spectra, key=get_wavelength_at_peak_intensity)

    @cached_property
    def fp_spectra(self) -> list[Spectrum]:
        """Aligned fluorophore emission spectra."""
        # fetch emission spectra
        fp_sp = [Spectrum.from_fpbase(fp_name) for fp_name in self.fp_names]
        
        # sort spectra by wavelength
        # fp_sp = self._sort_fp_spectra(fp_sp)

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
        fp_spectra = self.fp_spectra
        return [
            fp_spectrum.bin_intensity(self.n_bins, self.interval)
            for fp_spectrum in fp_spectra
        ]

    def _normalize(self, intensities: list[torch.Tensor]) -> list[torch.Tensor]:
        """Normalize the intensities of the emission spectra in [0, 1].
        
        Parameters
        ----------
        intensities : list[torch.Tensor]
            The intensities to normalize.
        
        Returns
        -------
        list[torch.Tensor]
            The normalized intensities.
        """
        return [
            (curr - curr.min()) / (curr.max() - curr.min())
            for curr in intensities
        ]
        
    def create(self, binned: bool = True, normalize: bool = True) -> torch.Tensor:
        """Create the reference matrix.
        
        The shape of the matrix is [W, F], where W is the number of wavelength bins
        and F is the number of fluorophores.
        
        Parameters
        ----------
        binned : bool
            Whether to use binned intensities. Default is True.
        normalize : bool
            Whether to normalize the intensities. Default is True.
        
        Returns
        -------
        torch.Tensor
            The reference matrix.
        """
        intensities = (
            self.binned_fp_intensities 
            if binned else [fp.intensity for fp in self.fp_spectra]
        )
        if normalize:
            intensities = self._normalize(intensities)
        return torch.stack(
            [intensity for intensity in intensities],
            axis=1
        )
