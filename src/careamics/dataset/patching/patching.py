"""Patching functions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from careamics.dataset.dataset_utils.synthetic_noise import SyntheticNoise
from ...utils.logging import get_logger
from ..dataset_utils import reshape_array
from ..dataset_utils.running_stats import compute_normalization_stats
from .sequential_patching import extract_patches_sequential

logger = get_logger(__name__)


@dataclass
class Stats:
    """Dataclass to store statistics."""

    means: Union[NDArray, tuple, list, None]
    """Mean of the data across channels."""

    stds: Union[NDArray, tuple, list, None]
    """Standard deviation of the data across channels."""

    def get_statistics(self) -> tuple[list[float], list[float]]:
        """Return the means and standard deviations.

        Returns
        -------
        tuple of two lists of floats
            Means and standard deviations.
        """
        if self.means is None or self.stds is None:
            return [], []

        return list(self.means), list(self.stds)


@dataclass
class PatchedOutput:
    """Dataclass to store patches and statistics."""

    patches: Union[NDArray]
    """Image patches."""

    targets: Union[NDArray, None]
    """Target patches."""

    image_stats: Stats
    """Statistics of the image patches."""

    target_stats: Stats
    """Statistics of the target patches."""


# called by in memory dataset
def prepare_patches_supervised(
    train_files: list[Path],
    target_files: list[Path],
    axes: str,
    patch_size: Union[list[int], tuple[int, ...]],
    read_source_func: Callable,
    read_source_kwargs: Optional[dict[str, Any]],
    norm_strategy: Literal["channel_wise", "global"],
    gaussian_noise_factor: Optional[float] = None,
    poisson_noise_factor: Optional[float] = None
) -> PatchedOutput:
    """
    Iterate over data source and create an array of patches and corresponding targets.

    The lists of Paths should be pre-sorted.

    Parameters
    ----------
    train_files : list of pathlib.Path
        List of paths to training data.
    target_files : list of pathlib.Path
        List of paths to target data.
    axes : str
        Axes of the data.
    patch_size : list or tuple of int
        Size of the patches.
    read_source_func : Callable
        Function to read the data.
    read_source_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the read_source_func.
    norm_strategy : Literal["channel_wise", "global"]
        Normalization strategy.
    gaussian_noise_factor : Optional[float]
        A factor determining the magnitude of Gaussian noise. Specifically, Gaussian
        noise is drawn from `N(0, gaussian_noise_factor * data_std)`, where data_std is
        the standard deviation of the input data. If None, Gaussian noise is disabled.
        Default is None.
    poisson_noise_factor : Optional[float]
        A factor determining the magnitude of Poisson noise. Specifically, resulting
        poisson noise will be proportional to `sqrt(I/poisson_noise_factor)`, where `I`
        is the pixel-wise intensity of the input image. If None, Poisson noise is
        disabled. Default is None.

    Returns
    -------
    np.ndarray
        Array of patches.
    """
    if read_source_kwargs is None:
        read_source_kwargs = {}
    means, stds, num_samples = 0, 0, 0
    all_patches, all_targets = [], []
    for train_filename, target_filename in zip(train_files, target_files):
        try:
            sample: np.ndarray = read_source_func(train_filename, **read_source_kwargs)
            target: np.ndarray = read_source_func(target_filename, **read_source_kwargs)
            means += sample.mean()
            stds += sample.std()
            num_samples += 1

            # reshape array
            sample = reshape_array(sample, axes)
            target = reshape_array(target, axes)
            
            # apply synthetic noise (if required)
            synthetic_noise = SyntheticNoise(
                poisson_noise_factor, gaussian_noise_factor * sample.std()
            )
            sample, target = synthetic_noise(sample, target)

            # generate patches, return a generator
            patches, targets = extract_patches_sequential(
                sample, patch_size=patch_size, target=target
            )

            # convert generator to list and add to all_patches
            all_patches.append(patches)

            # ensure targets are not None (type checking)
            if targets is not None:
                all_targets.append(targets)
            else:
                raise ValueError(f"No target found for {target_filename}.")

        except Exception as e:
            # emit warning and continue
            logger.error(f"Failed to read {train_filename} or {target_filename}: {e}")

    # raise error if no valid samples found
    if num_samples == 0 or len(all_patches) == 0:
        raise ValueError(
            f"No valid samples found in the input data: {train_files} and "
            f"{target_files}."
        )

    patch_array: np.ndarray = np.concatenate(all_patches, axis=0)
    target_array: np.ndarray = np.concatenate(all_targets, axis=0)
    logger.info(f"Extracted {patch_array.shape[0]} patches from input array.")
    
    image_means, image_stds = compute_normalization_stats(
        patch_array, norm_strategy
    )
    target_means, target_stds = compute_normalization_stats(
        target_array, norm_strategy
    )

    return PatchedOutput(
        patch_array,
        target_array,
        Stats(image_means, image_stds),
        Stats(target_means, target_stds),
    )


# called by in_memory_dataset
def prepare_patches_unsupervised(
    train_files: list[Path],
    axes: str,
    patch_size: Union[list[int], tuple[int]],
    read_source_func: Callable,
    read_source_kwargs: Optional[dict[str, Any]],
    norm_strategy: Literal["channel_wise", "global"],
    gaussian_noise_factor: Optional[float] = None,
    poisson_noise_factor: Optional[float] = None
) -> PatchedOutput:
    """Iterate over data source and create an array of patches.

    This method returns the mean and standard deviation of the image.

    Parameters
    ----------
    train_files : list of pathlib.Path
        List of paths to training data.
    axes : str
        Axes of the data.
    patch_size : list or tuple of int
        Size of the patches.
    read_source_func : Callable
        Function to read the data.
    read_source_kwargs : Optional[dict[str, Any]]
        Keyword arguments to pass to the read_source_func.
    norm_strategy : Literal["channel_wise", "global"]
        Normalization strategy.
    gaussian_noise_factor : Optional[float]
        A factor determining the magnitude of Gaussian noise. Specifically, Gaussian
        noise is drawn from `N(0, gaussian_noise_factor * data_std)`, where data_std is
        the standard deviation of the input data. If None, Gaussian noise is disabled.
        Default is None.
    poisson_noise_factor : Optional[float]
        A factor determining the magnitude of Poisson noise. Specifically, resulting
        poisson noise will be proportional to `sqrt(I/poisson_noise_factor)`, where `I`
        is the pixel-wise intensity of the input image. If None, Poisson noise is
        disabled. Default is None.

    Returns
    -------
    PatchedOutput
        Dataclass holding patches and their statistics.
    """
    if read_source_kwargs is None:
        read_source_kwargs = {}
    means, stds, num_samples = 0, 0, 0
    all_patches = []
    for filename in tqdm(train_files, desc="Reading files"):
        try:
            sample: np.ndarray = read_source_func(filename, **read_source_kwargs)
            means += sample.mean() # TODO: what do we need this for?
            stds += sample.std() # TODO: what do we need this for?
            num_samples += 1

            # reshape array
            sample = reshape_array(sample, axes)
            
            # apply synthetic noise (if required)
            synthetic_noise = SyntheticNoise(
                poisson_noise_factor, gaussian_noise_factor * sample.std()
            )
            sample = synthetic_noise(sample)
            
            # generate patches, return a generator
            patches, _ = extract_patches_sequential(sample, patch_size=patch_size)

            # convert generator to list and add to all_patches
            all_patches.append(patches)
        except Exception as e:
            # emit warning and continue
            logger.error(f"Failed to read {filename}: {e}")

    # raise error if no valid samples found
    if num_samples == 0:
        raise ValueError(f"No valid samples found in the input data: {train_files}.")

    patch_array: np.ndarray = np.concatenate(all_patches)
    logger.info(f"Extracted {patch_array.shape[0]} patches from input array.")
    
    image_means, image_stds = compute_normalization_stats(
        patch_array, norm_strategy
    )
    
    return PatchedOutput(
        patch_array, None, Stats(image_means, image_stds), Stats((), ())
    )


# called on arrays by in memory dataset
def prepare_patches_supervised_array(
    data: NDArray,
    axes: str,
    data_target: NDArray,
    patch_size: Union[list[int], tuple[int]],
    norm_strategy: Literal["channel_wise", "global"],
    gaussian_noise_factor: Optional[float] = None,
    poisson_noise_factor: Optional[float] = None
) -> PatchedOutput:
    """Iterate over data source and create an array of patches.

    This method expects an array of shape SC(Z)YX, where S and C can be singleton
    dimensions.

    Patches returned are of shape SC(Z)YX, where S is now the patches dimension.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array.
    axes : str
        Axes of the data.
    data_target : numpy.ndarray
        Target data array.
    patch_size : list or tuple of int
        Size of the patches.
    norm_strategy : Literal["channel_wise", "global"]
        Normalization strategy.
    gaussian_noise_factor : Optional[float]
        A factor determining the magnitude of Gaussian noise. Specifically, Gaussian
        noise is drawn from `N(0, gaussian_noise_factor * data_std)`, where data_std is
        the standard deviation of the input data. If None, Gaussian noise is disabled.
        Default is None.
    poisson_noise_factor : Optional[float]
        A factor determining the magnitude of Poisson noise. Specifically, resulting
        poisson noise will be proportional to `sqrt(I/poisson_noise_factor)`, where `I`
        is the pixel-wise intensity of the input image. If None, Poisson noise is
        disabled. Default is None.

    Returns
    -------
    PatchedOutput
        Dataclass holding the source and target patches, with their statistics.
    """
    # reshape array
    reshaped_sample = reshape_array(data, axes)
    reshaped_target = reshape_array(data_target, axes)

    # compute statistics
    image_means, image_stds = compute_normalization_stats(
        reshaped_sample, norm_strategy
    )
    target_means, target_stds = compute_normalization_stats(
        reshaped_target, norm_strategy
    )
    
    # apply synthetic noise (if required)
    synthetic_noise = SyntheticNoise(
        poisson_noise_factor, gaussian_noise_factor * image_stds
    )
    reshaped_sample, reshaped_target = synthetic_noise(
        reshaped_sample, reshaped_target
    )

    # generate patches, return a generator
    patches, patch_targets = extract_patches_sequential(
        reshaped_sample, patch_size=patch_size, target=reshaped_target
    )

    if patch_targets is None:
        raise ValueError("No target extracted.")

    logger.info(f"Extracted {patches.shape[0]} patches from input array.")

    return PatchedOutput(
        patches,
        patch_targets,
        Stats(image_means, image_stds),
        Stats(target_means, target_stds),
    )


# called by in memory dataset
def prepare_patches_unsupervised_array(
    data: NDArray,
    axes: str,
    patch_size: Union[list[int], tuple[int]],
    norm_strategy: Literal["channel_wise", "global"],
    gaussian_noise_factor: Optional[float] = None,
    poisson_noise_factor: Optional[float] = None
) -> PatchedOutput:
    """
    Iterate over data source and create an array of patches.

    This method expects an array of shape SC(Z)YX, where S and C can be singleton
    dimensions.

    Patches returned are of shape SC(Z)YX, where S is now the patches dimension.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array.
    axes : str
        Axes of the data.
    patch_size : list or tuple of int
        Size of the patches.
    norm_strategy : Literal["channel_wise", "global"]
        Normalization strategy.
    gaussian_noise_factor : Optional[float]
        A factor determining the magnitude of Gaussian noise. Specifically, Gaussian
        noise is drawn from `N(0, gaussian_noise_factor * data_std)`, where data_std is
        the standard deviation of the input data. If None, Gaussian noise is disabled.
        Default is None.
    poisson_noise_factor : Optional[float]
        A factor determining the magnitude of Poisson noise. Specifically, resulting
        poisson noise will be proportional to `sqrt(I/poisson_noise_factor)`, where `I`
        is the pixel-wise intensity of the input image. If None, Poisson noise is
        disabled. Default is None.
    

    Returns
    -------
    PatchedOutput
        Dataclass holding the patches and their statistics.
    """
    # reshape array
    reshaped_sample = reshape_array(data, axes)

    # calculate mean and std
    means, stds = compute_normalization_stats(reshaped_sample, norm_strategy)
    
    # apply synthetic noise (if required)
    synthetic_noise = SyntheticNoise(
        poisson_noise_factor, gaussian_noise_factor * stds
    )
    reshaped_sample = synthetic_noise(reshaped_sample)

    # generate patches, return a generator
    patches, _ = extract_patches_sequential(reshaped_sample, patch_size=patch_size)

    return PatchedOutput(patches, None, Stats(means, stds), Stats((), ()))
