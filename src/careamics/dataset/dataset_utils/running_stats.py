"""Computing data statistics."""
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .dataset_utils import Stats


def _compute_min_max_stats(
    data: NDArray, 
    strategy: Literal["channel-wise", "global"] = "channel-wise"
) -> tuple[NDArray, NDArray]:
    """
    Compute min and max of an array.
    
    For robustness, the min and max values are taken as the 1% and 99% percentiles
    of the intensity values in the data.

    Parameters
    ----------
    data : NDArray
        Input data array. Expected input shape is (S, C, (Z), Y, X).
    norm_strategy : Literal["channel-wise", "global"]
        Normalization strategy. Default is "channel-wise".

    Returns
    -------
    tuple of (list of floats, list of floats)
        Lists of min and max values per channel.
    """
    if strategy == "channel-wise":
        # Define the list of axes excluding the channel axis
        axes = tuple(np.delete(np.arange(data.ndim), 1))
        stats = (
            np.quantile(data, 0.005, axis=axes), # (C,)
            np.quantile(data, 0.995, axis=axes) # (C,)
        )
    elif strategy == "global":
        axes = tuple(np.arange(data.ndim))
        stats = (
            np.asarray(np.quantile(data, 0.005, axis=axes))[None], # (1,) 
            np.asarray(np.quantile(data, 0.995, axis=axes))[None] # (1,)
        )
    else:
        raise ValueError(
            (
                f"Unknown normalization strategy: {strategy}."
                "Available ones are 'channel-wise' and 'global'."
            )
        )
    return stats


def _compute_mean_std_stats(
    data: NDArray, 
    strategy: Literal["channel-wise", "global"] = "channel-wise"
) -> tuple[NDArray, NDArray]:
    """
    Compute mean and standard deviation of an array.

    Parameters
    ----------
    data : NDArray
        Input data array. Expected input shape is (S, C, (Z), Y, X)
    norm_strategy : Literal["channel-wise", "global"]
        Normalization strategy. Default is "channel-wise".

    Returns
    -------
    tuple of (list of floats, list of floats)
        Lists of mean and standard deviation values per channel.
    """
    if strategy == "channel-wise":
        # Define the list of axes excluding the channel axis
        axes = tuple(np.delete(np.arange(data.ndim), 1))
        stats = (
            np.mean(data, axis=axes), # (C,)
            np.std(data, axis=axes) # (C,)
        )
    elif strategy == "global":
        axes = tuple(np.arange(data.ndim))
        stats = (
            np.asarray(np.mean(data, axis=axes))[None], # (1,)
            np.asarray(np.std(data, axis=axes))[None] # (1,)
        )
    else:
        raise ValueError(
            (
                f"Unknown normalization strategy: {strategy}."
                "Available ones are 'channel-wise' and 'global'."
            )
        )
    return stats


def compute_normalization_stats(
    data: NDArray,
    method: Literal["normalize", "standardize"],
    strategy: Literal["channel-wise", "global"] = "channel-wise",
) -> Stats:
    """Compute normalization statistics on the input array.
    
    Parameters
    ----------
    image : NDArray
        Input array. Expected input shape is (S, C, (Z), Y, X).
    norm_type : Literal["normalize", "standardize"]
        Normalization type.
    strategy : Literal["channel-wise", "global"]
        Normalization strategy. Default is "channel-wise".
        
    Returns
    -------
    Stats
        Normalization statistics.
    """
    stats = {"means": None, "stds": None, "mins": None, "maxs": None}
    if method == "normalize":
        stats["mins"], stats["maxs"] = _compute_min_max_stats(data, strategy)
    elif method == "standardize":
        stats["means"], stats["stds"] = _compute_mean_std_stats(data, strategy)
    else:
        raise ValueError(
            (
                f"Unknown normalization type: {method}."
                "Available ones are 'normalize' and 'standardize'."
            )
        )
    return Stats(**stats)


def update_iterative_stats(
    count: NDArray, mean: NDArray, m2: NDArray, new_values: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """Update the mean and variance of an array iteratively.

    Parameters
    ----------
    count : NDArray
        Number of elements in the array. Shape: (C,).
    mean : NDArray
        Mean of the array. Shape: (C,).
    m2 : NDArray
        Variance of the array. Shape: (C,).
    new_values : NDArray
        New values to add to the mean and variance. Shape: (C, 1, 1, Z, Y, X).

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        Updated count, mean, and variance.
    """
    num_channels = len(new_values)

    # --- update channel-wise counts ---
    count += np.ones_like(count) * np.prod(new_values.shape[1:])

    # --- update channel-wise mean ---
    # compute (new_values - old_mean) -> shape: (C, Z*Y*X)
    delta = new_values.reshape(num_channels, -1) - mean.reshape(num_channels, 1)
    mean += np.sum(delta / count.reshape(num_channels, 1), axis=1)

    # --- update channel-wise SoS ---
    # compute (new_values - new_mean) -> shape: (C, Z*Y*X)
    delta2 = new_values.reshape(num_channels, -1) - mean.reshape(num_channels, 1)
    m2 += np.sum(delta * delta2, axis=1)

    return count, mean, m2


def finalize_iterative_stats(
    count: NDArray, mean: NDArray, m2: NDArray
) -> tuple[NDArray, NDArray]:
    """Finalize the mean and variance computation.

    Parameters
    ----------
    count : NDArray
        Number of elements in the array. Shape: (C,).
    mean : NDArray
        Mean of the array. Shape: (C,).
    m2 : NDArray
        Variance of the array. Shape: (C,).

    Returns
    -------
    tuple[NDArray, NDArray]
        Final channel-wise mean and standard deviation.
    """
    std = np.sqrt(m2 / count)
    if any(c < 2 for c in count):
        return np.full(mean.shape, np.nan), np.full(std.shape, np.nan)
    else:
        return mean, std


class WelfordStatistics:
    """Compute Welford statistics iteratively.

    The Welford algorithm is used to compute the mean and variance of an array
    iteratively. Based on the implementation from:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def update(self, array: NDArray, sample_idx: int) -> None:
        """Update the Welford statistics.

        Parameters
        ----------
        array : NDArray
            Input array.
        sample_idx : int
            Current sample number.
        """
        self.sample_idx = sample_idx
        sample_channels = np.array(np.split(array, array.shape[1], axis=1))

        # Initialize the statistics
        if self.sample_idx == 0:
            # Compute the mean and standard deviation
            self.mean, _ = compute_normalization_stats(array)
            # Initialize the count and m2 with zero-valued arrays of shape (C,)
            self.count, self.mean, self.m2 = update_iterative_stats(
                count=np.zeros(array.shape[1]),
                mean=self.mean,
                m2=np.zeros(array.shape[1]),
                new_values=sample_channels,
            )
        else:
            # Update the statistics
            self.count, self.mean, self.m2 = update_iterative_stats(
                count=self.count, mean=self.mean, m2=self.m2, new_values=sample_channels
            )

        self.sample_idx += 1

    def finalize(self) -> tuple[NDArray, NDArray]:
        """Finalize the Welford statistics.

        Returns
        -------
        tuple or numpy arrays
            Final mean and standard deviation.
        """
        return finalize_iterative_stats(self.count, self.mean, self.m2)


class RunningMinMaxStatistics:
    """Compute running min and max statistics.
    
    Min and max statistics are computed iteratively and updated for each input array.
    
    For robustness, the min and max values are taken as the 1% and 99% percentiles
    of the intensity values in the data.
    """
    
    def __init__(self) -> None:
        self.mins = None
        self.maxs = None
    
    def update(self, array: NDArray) -> None:
        """Update the running min and max statistics.
        
        Parameters
        ----------
        array : NDArray
            Input array of shape (S, C, (Z), Y, X).
        """
        # TODO: make quantiles as a parameter!
        axes = tuple(np.delete(np.arange(array.ndim), 1))
        if self.mins is None:
            self.mins = np.quantile(array, 0.005, axis=axes) # (C,)
            self.maxs = np.quantile(array, 0.995, axis=axes) # (C,)
        else:
            self.mins = np.minimum(
                self.mins, np.quantile(array, 0.005, axis=axes)
            ) # (C,)
            self.maxs = np.maximum(
                self.maxs, np.quantile(array, 0.995, axis=axes)
            ) # (C,)


# from multiprocessing import Value
# from typing import tuple

# import numpy as np


# class RunningStats:
#     """Calculates running mean and std."""

#     def __init__(self) -> None:
#         self.reset()

#     def reset(self) -> None:
#         """Reset the running stats."""
#         self.avg_mean = Value("d", 0)
#         self.avg_std = Value("d", 0)
#         self.m2 = Value("d", 0)
#         self.count = Value("i", 0)

#     def init(self, mean: float, std: float) -> None:
#         """Initialize running stats."""
#         with self.avg_mean.get_lock():
#             self.avg_mean.value += mean
#         with self.avg_std.get_lock():
#             self.avg_std.value = std

#     def compute_std(self) -> tuple[float, float]:
#         """Compute std."""
#         if self.count.value >= 2:
#             self.avg_std.value = np.sqrt(self.m2.value / self.count.value)

#     def update(self, value: float) -> None:
#         """Update running stats."""
#         with self.count.get_lock():
#             self.count.value += 1
#         delta = value - self.avg_mean.value
#         with self.avg_mean.get_lock():
#             self.avg_mean.value += delta / self.count.value
#         delta2 = value - self.avg_mean.value
#         with self.m2.get_lock():
#             self.m2.value += delta * delta2
