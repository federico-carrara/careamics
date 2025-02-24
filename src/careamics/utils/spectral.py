"""Utility functions for working with spectral data."""

import warnings
from functools import cached_property
from typing import Literal, Optional, Sequence, Union

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .fpbase import get_fp_emission_spectrum


def _get_bins(num_bins: int, interval: Sequence[int],) -> torch.Tensor:
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
        bin_edges = _get_bins(
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
    
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    fp_names: Sequence[str]
    """The names of the fluorophores to include in the reference matrix."""
    
    n_bins: int = 32
    """The number of wavelength bins to use for the FP spectra."""
    
    interval: Optional[Sequence[int]] = None
    """The interval of wavelengths in which binning is done. Wavelengths outside this
    interval are ignored. If `None`, the interval is set to the range of the wavelength."""
    
    matrix: Optional[torch.Tensor] = None
    """The reference matrix containing the fluorophore emission spectra."""
    
    @classmethod
    def from_array(
        cls,
        matrix: NDArray,
        fp_names: Sequence[str],
        n_bins: int,
        interval: Sequence[int],
        background: bool = False,
    ) -> "FPRefMatrix":
        """Create a reference matrix from an array of fluorophore emission spectra.
        
        Parameters
        ----------
        matrix : NDArray
            The array containing the fluorophore emission spectra, of shape [W, F].
        fp_names : Sequence[str]
            The names of the fluorophores.
        n_bins : int
            The number of wavelength bins to use for the FP spectra.
        interval : Sequence[int]
            The interval of wavelengths in which binning is done. Wavelengths outside
            this interval are ignored.
        background : bool
            Whether the array contains a background spectrum. Default is False.
        """
        F = matrix.shape[1]
        if background:
            assert F == len(fp_names) + 1, (
                "Number of fluorophores and background spectrum do not match!"
            )
        else:
            assert F == len(fp_names), (
                "Number of fluorophores and background spectrum do not match!"
            )

        return cls(
            fp_names=fp_names,
            n_bins=n_bins,
            interval=interval,
            matrix=torch.tensor(matrix),
        )
    
    def _fetch_fp_spectra(self) -> list[Spectrum]:
        """Fetch the fluorophore emission spectra from FPbase."""
        return [Spectrum.from_fpbase(fp_name) for fp_name in self.fp_names]
    
    def _sort_fp_spectra(self, fp_spectra: list[Spectrum]) -> list[Spectrum]:
        """Sort the fluorophore emission spectra by wavelength."""
        def get_wavelength_at_peak_intensity(sp: Spectrum) -> float:
            return sp.wavelength[sp.intensity.argmax()]

        sorted_spectra, sorted_fp_names = zip(
            *sorted(
                zip(fp_spectra, self.fp_names), 
                key=lambda x: get_wavelength_at_peak_intensity(x[0])
            )
        )
        self.fp_names = sorted_fp_names
        return list(sorted_spectra)
    
    def _align_fp_spectra(self, fp_spectra: list[Spectrum]) -> list[Spectrum]:
        """Align the fluorophore emission spectra on the same wavelength grid.
        
        Parameters
        ----------
        fp_spectra : list[Spectrum]
            The fluorophore emission spectra to align.
        
        Returns
        -------
        list[Spectrum]
            The aligned fluorophore emission spectra.
        """
        fp_sp0, *rest = fp_spectra
        aligned_fp_sp = []
        for sp in rest:
            fp_sp0, sp = fp_sp0._align(sp)
            aligned_fp_sp.append(sp)
        aligned_fp_sp = [fp_sp0] + aligned_fp_sp
        return aligned_fp_sp

    @cached_property
    def fp_spectra(self) -> list[Spectrum]:
        """Aligned fluorophore emission spectra."""
        # fetch emission spectra
        fp_sp = self._fetch_fp_spectra()
        
        # sort spectra by wavelength
        fp_sp = self._sort_fp_spectra(fp_sp)

        # align spectra
        return self._align_fp_spectra(fp_sp)

    def _bin_fp_intensities(self, spectra: list[Spectrum]) -> list[torch.Tensor]:
        """Binned fluorophore emission spectrum intensities."""
        return [
            spectrum.bin_intensity(self.n_bins, self.interval)
            for spectrum in spectra
        ]

    @staticmethod
    def _normalize(intensity: torch.Tensor) -> torch.Tensor:
        """Normalize emission spectra intensity s.t. the integral sums up to 1.
        
        Parameters
        ----------
        intensities : torch.Tensor
            The spectrum intensity to normalize.
        
        Returns
        -------
        torch.Tensor
            The normalized spectrum intensity.
        """
        # TODO: also scale by QE and those things?
        return intensity / intensity.sum()
    
    def create(self, binned: bool = True, normalize: bool = True) -> torch.Tensor:
        """The matrix containing the fluorophore reference spectra.
        
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
        # get the intensities of each fluorophore spectrum
        if binned:
            intensities = self._bin_fp_intensities(self.fp_spectra) 
        else:
            intensities = [fp_spectrum.intensity for fp_spectrum in self.fp_spectra]
        
        # normalize the intensities
        if normalize:
            intensities = [self._normalize(intensity) for intensity in intensities]
        
        self.matrix = torch.stack([intensity for intensity in intensities], axis=1)
        return self.matrix

    def _get_spectrum_from_image(
        self,
        image: Union[np.ndarray, torch.Tensor],
        coords: tuple[int, ...],
        n_pixels: int = 5
    ) -> torch.Tensor:
        """Get spectrum taken from a patch of an image.
        
        Parameters
        ----------
        image : Union[np.ndarray, torch.Tensor]
            The image from which to extract spectrum. 
            Shape should be (C, [Z], Y, X).
        coords : tuple[int, ...]
            The coordinates of the spectrum to extract from the image, in the form 
            ([Z], Y, X). A square region of size [`2*n_pixels`x`2*n_pixels`] centered
            at (Y, X) is used to compute the spectrum.
        n_pixels : int
            The size of the square region around the coordinates at which the
            spectrum is extracted. Default is 5.
        
        Returns
        -------
        torch.Tensor
            The obtained spectrum.
        """
        spatial_ndims = image.ndim - 1
        assert spatial_ndims in [2, 3], "Image must either 2D or 3D!"
        assert len(coords) == spatial_ndims, (
            f"Coordinates should be as many as the image spatial dimensions! "
            f"Got instead {len(coords)} coordinates for image of shape {image.shape}."
        )
        
        # extract background spectrum
        bg_yx_slices = [
            slice(coords[-2]-n_pixels, coords[-2]+n_pixels),
            slice(coords[-1]-n_pixels, coords[-1]+n_pixels),
        ]
        if spatial_ndims == 2:
            bg_spectrum_intensity = image[:, bg_yx_slices[0], bg_yx_slices[1]]
        elif spatial_ndims == 3:
            bg_spectrum_intensity = image[
                :, coords[0], bg_yx_slices[0], bg_yx_slices[1]
            ]
        bg_spectrum_intensity: np.ndarray = np.asarray(bg_spectrum_intensity)
        bg_spectrum_intensity = bg_spectrum_intensity.mean(axis=(-1, -2))
        bg_spectrum_intensity: torch.Tensor = torch.tensor(
            bg_spectrum_intensity, dtype=torch.float32
        )
        
        # normalize background spectrum
        return bg_spectrum_intensity
        
    def add_background_spectrum(
        self,
        method: Literal["random", "constant", "from_image"],
        image: Optional[Union[np.ndarray, torch.Tensor]] = None,
        coords: Optional[tuple[int, ...]] = None,
        n_pixels: int = 5
    ) -> torch.Tensor:
        """Add a (background) spectrum to the reference matrix.
        
        If `image` is provided, the background spectrum is extracted from the image
        at the given coordinates. Otherwise, the background spectrum is randomly
        initialized from a uniform distribution or set to a constant value.
        
        Parameters
        ----------
        method : Literal["random", "constant", "from_image"]
            The method to use for adding the background spectrum.
        image : Optional[Union[np.ndarray, torch.Tensor]]
            The image from which to extract the background spectrum. 
            Shape should be (C, [Z], Y, X).Default is None.
        coords : Opional[tuple[int, ...]]
            The coordinates of the spectrum to extract from the image, in the form 
            ([Z], Y, X). A square region of size [`2*n_pixels`x`2*n_pixels`] centered
            at (Y, X) is used to compute the spectrum. Default is None.
        n_pixels : int
            The size of the square region around the coordinates at which the
            spectrum is extracted. Default is 5.
        
        Returns
        -------
        torch.Tensor
            The updated reference matrix.
        """
        assert self.matrix.shape[1] <= len(self.fp_names), (
            "Background spectrum already added!"
        )
        if method == "from image":
            assert image is not None, "Image must be provided to extract background spectrum!"
            assert coords is not None, "Coordinates must be provided to extract background spectrum!"
        
        # get background spectrum
        if method == "from_image":
            bg_spectrum_intensity = self._get_spectrum_from_image(
                image, coords, n_pixels
            )
        elif method == "random":
            bg_spectrum_intensity = torch.rand(self.n_bins)
        elif method == "constant":
            bg_spectrum_intensity = torch.ones(self.n_bins)
        else:
            raise ValueError(
                f"Unknown method {method} for adding background spectrum!"
                "Choose from 'random', 'constant', 'from_image'."
            )
        
        # normalize intensity
        bg_spectrum_intensity = self._normalize(bg_spectrum_intensity)
        
        # add background spectrum to the reference matrix
        self.matrix = torch.cat([self.matrix, bg_spectrum_intensity.unsqueeze(1)], axis=1)
        return self.matrix
    
    def plot(self, show_wavelengths: bool = True) -> None:
        """Plot the reference matrix.
        
        Parameters
        ----------
        show_wavelengths : bool
            Whether to show the wavelength values on the x-axis. Default is False.
        """
        assert hasattr(self, "matrix"), "Reference matrix not created yet!"
        if show_wavelengths:
            assert self.interval is not None, "Wavelength interval not provided!"
            bins = _get_bins(self.n_bins, self.interval)
            bins = np.asarray(bins)
            bin_centers = (bins[:-1] + bins[1:]) / 2
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = list(self.fp_names) + ["background"]
        for i in range(self.matrix.shape[1]):
            ax.plot(self.matrix[:, i], label=labels[i])
        ax.set_xlabel("Wavelength bins")
        ax.set_ylabel("Normalized intensity")
        ax.set_title("Reference FP spectra")
        ax.set_xticks(range(self.n_bins))
        ax.set_xticklabels(bin_centers.astype(int), rotation=45)
        ax.legend()
        plt.show()