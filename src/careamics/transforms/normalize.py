"""Normalization and denormalization transforms for image patches."""

from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

from careamics.transforms.transform import Transform


def _reshape_stats(stats: list[float], ndim: int) -> NDArray:
    """Reshape stats to match the number of dimensions of the input image.

    This allows to broadcast the stats (mean or std) to the image dimensions, and
    thus directly perform a vectorial calculation.

    Parameters
    ----------
    stats : list of float
        List of stats, mean or standard deviation.
    ndim : int
        Number of dimensions of the image, including the C channel.

    Returns
    -------
    NDArray
        Reshaped stats.
    """
    arr = np.array(stats)
    if arr.ndim == 1:
        return arr[(..., *[np.newaxis] * (ndim - 1))]
    elif arr.ndim == ndim:
        return arr
    elif arr.ndim <= ndim:
        return arr[(..., *np.newaxis * (ndim - arr.ndim))]
    else:
        raise ValueError(
            f"Stats array has too many dimensions ({arr.ndim}) compared to the image "
            f"({ndim})."
        )

class Normalize(Transform):
    """
    Normalize an image or image patch.

    Normalization is uses min-max to normalize the data in [0, 1].
    This transform expects C(Z)YX dimensions.

    Not that it returns a float32 image.

    Parameters
    ----------
    image_mins : list[float]
        Minimum value per channel.
    image_maxs : list[float]
        Maximum value per channel.
    target_mins : list[float], optional
        Target minimum value per channel, by default None.
    target_maxs : list[float], optional
        Target maximum value per channel, by default None.
    """

    def __init__(
        self,
        image_mins: list[float],
        image_maxs: list[float],
        target_mins: Optional[list[float]] = None,
        target_maxs: Optional[list[float]] = None,
        strategy: Literal["channel-wise", "global"] = "channel-wise",
    ):
        """Constructor.

        Parameters
        ----------
        image_mins : list[float]
            Minimum value per channel.
        image_maxs : list[float]
            Maximum value per channel.
        target_mins : list[float], optional
            Target minimum value per channel, by default None.
        target_maxs : list[float], optional
            Target maximum value per channel, by default None.
        """
        self.image_mins = image_mins
        self.image_maxs = image_maxs
        self.target_mins = target_mins
        self.target_maxs = target_maxs
        self.strategy = strategy

        self.eps = 1e-6

    def __call__(
        self,
        patch: np.ndarray,
        target: Optional[NDArray] = None,
        **additional_arrays: NDArray,
    ) -> tuple[NDArray, Optional[NDArray], dict[str, NDArray]]:
        """Apply the transform to the source patch and the target (optional).

        Parameters
        ----------
        patch : NDArray
            Patch, 2D or 3D, shape C(Z)YX.
        target : NDArray, optional
            Target for the patch, by default None.
        **additional_arrays : NDArray
            Additional arrays that will be transformed identically to `patch` and
            `target`.

        Returns
        -------
        tuple of NDArray
            Transformed patch and target, the target can be returned as `None`.
        """
        if self.strategy == "channel-wise" and len(self.image_mins) != patch.shape[0]:
            raise ValueError(
                f"Number of mins (got a list of size {len(self.image_mins)}) and "
                f"number of channels (got shape {patch.shape} for C(Z)YX) do not match."
            )
            # TODO: patch can also be of shape SC(Z)YX, e.g., in the case we call dset[:S].
            # In that case this check will fail.
        if len(additional_arrays) != 0:
            raise NotImplementedError(
                "Transforming additional arrays is currently not supported for "
                "`Normalize`."
            )

        # reshape mins and maxs and apply the normalization to the patch
        means = _reshape_stats(self.image_mins, patch.ndim)
        stds = _reshape_stats(self.image_maxs, patch.ndim)
        norm_patch = self._apply(patch, means, stds)

        # same for the target patch
        if (
            target is not None
            and self.target_mins is not None
            and self.target_maxs is not None
        ):
            target_mins = _reshape_stats(self.target_mins, target.ndim)
            target_maxs = _reshape_stats(self.target_maxs, target.ndim)
            norm_target = self._apply(target, target_mins, target_maxs)
        else:
            norm_target = None

        return norm_patch, norm_target, additional_arrays

    def _apply(self, patch: NDArray, min_: NDArray, max_: NDArray) -> NDArray:
        """
        Apply the transform to the image.

        Parameters
        ----------
        patch : NDArray
            Image patch, 2D or 3D, shape C(Z)YX.
        min_ : NDArray
            Minimum values.
        max_ : NDArray
            Maximum values.

        Returns
        -------
        NDArray
            Standardized image patch.
        """
        return ((patch - min_) / (max_ - min_)).astype(np.float32)


class Denormalize:
    """
    Denormalize an image.

    Denormalization is performed expecting a zero mean and unit variance input. This
    transform expects C(Z)YX dimensions.
    
    Parameters
    ----------
    image_mins : list[float]
        Minimum value per channel.
    image_maxs : list[float]
        Maximum value per channel.
    """

    def __init__(
        self,
        image_mins: list[float],
        image_maxs: list[float],
    ):
        """Constructor.

        Parameters
        ----------
        image_mins : list[float]
            Minimum value per channel.
        image_maxs : list[float]
            Maximum value per channel.
        """
        self.image_mins = image_mins
        self.image_maxs = image_maxs

    def __call__(self, patch: NDArray) -> NDArray:
        """Reverse the normalization operation for a batch of patches.

        Parameters
        ----------
        patch : NDArray
            Patch, 2D or 3D, shape BC(Z)YX.

        Returns
        -------
        NDArray
            Transformed array.
        """
        if len(self.image_mins) != patch.shape[1]:
            raise ValueError(
                f"Number of mins (got a list of size {len(self.image_mins)}) and "
                f"number of channels (got shape {patch.shape} for BC(Z)YX) do not "
                f"match."
            )

        mins = _reshape_stats(self.image_mins, patch.ndim)
        maxs = _reshape_stats(self.image_maxs, patch.ndim)

        denorm_array = self._apply(
            patch,
            np.swapaxes(mins, 0, 1),  # swap axes as C channel is axis 1
            np.swapaxes(maxs, 0, 1),
        )

        return denorm_array.astype(np.float32)

    def _apply(self, array: NDArray, mins: NDArray, maxs: NDArray) -> NDArray:
        """
        Apply the transform to the image.

        Parameters
        ----------
        array : NDArray
            Image patch, 2D or 3D, shape C(Z)YX.
        mins : NDArray
            Minimum values.
        maxs : NDArray
            Maximum values.

        Returns
        -------
        NDArray
            Denormalized image array.
        """
        return array * (maxs - mins) + mins