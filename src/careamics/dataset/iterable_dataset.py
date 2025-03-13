"""Iterable dataset used to load data file by file."""

from __future__ import annotations

import copy
from collections.abc import Generator
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from tqdm import tqdm
from torch.utils.data import IterableDataset

from careamics.config import DataConfig
from careamics.config.transformations import NormalizeModel, StandardizeModel
from careamics.file_io.read import read_tiff
from careamics.transforms import Compose

from ..utils.logging import get_logger
from .dataset_utils import iterate_over_files
from .dataset_utils.running_stats import RunningMinMaxStatistics, WelfordStatistics
from .dataset_utils.dataset_utils import Stats
from .patching.random_patching import extract_patches_random

logger = get_logger(__name__)


class PathIterableDataset(IterableDataset):
    """
    Dataset allowing extracting patches w/o loading whole data into memory.

    Parameters
    ----------
    data_config : DataConfig
        Data configuration.
    src_files : list of pathlib.Path
        List of data files.
    target_files : list of pathlib.Path, optional
        Optional list of target files, by default None.
    read_source_func : Callable, optional
        Read source function for custom types, by default read_tiff.
    read_source_kwargs : dict[str, Any], optional
        Additional keyword arguments for the read function, by default None.

    Attributes
    ----------
    data_path : list of pathlib.Path
        Path to the data, must be a directory.
    axes : str
        Description of axes in format STCZYX.
    """

    def __init__(
        self,
        data_config: DataConfig,
        src_files: list[Path],
        target_files: Optional[list[Path]] = None,
        read_source_func: Callable = read_tiff,
        read_source_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Constructors.

        Parameters
        ----------
        data_config : DataConfig
            Data configuration.
        src_files : list[Path]
            List of data files.
        target_files : list[Path] or None, optional
            Optional list of target files, by default None.
        read_source_func : Callable, optional
            Read source function for custom types, by default read_tiff.
        read_source_kwargs : dict[str, Any], optional
            Additional keyword arguments for the read function, by default None.
        """
        self.data_config = data_config
        self.data_files = src_files
        self.target_files = target_files
        self.read_source_func = read_source_func
        self.read_source_kwargs = read_source_kwargs
        self.norm_type = self.data_config.norm_type
        # FIXME: `norm_strategy` not defined here...

        # compute mean and std over the dataset
        self._set_image_stats()

        # create transform composed of normalization and other transforms
        if self.norm_type == "normalize":
            norm_transform = NormalizeModel(
                image_mins=self.image_stats.mins,
                image_maxs=self.image_stats.maxs,
                target_mins=self.target_stats.mins,
                target_maxs=self.target_stats.maxs,
            )
        elif self.norm_type == "standardize":
            norm_transform = StandardizeModel(
                image_means=self.image_stats.means,
                image_stds=self.image_stats.stds,
                target_means=self.target_stats.means,
                target_stds=self.target_stats.stds,
            )
            
        self.patch_transform = Compose(
            transform_list=[norm_transform] + data_config.transforms
        )
    
    def _calculate_mean_and_std(self) -> tuple[Stats, Stats]:
        """Calculate channel-wise mean and std of the dataset.

        Returns
        -------
        tuple[Stats, Stats]
            Data classes containing the image and target statistics.
        """
        num_samples = 0
        image_stats = WelfordStatistics()
        if self.target_files is not None:
            target_stats = WelfordStatistics()

        for sample, target, _ in tqdm(
            iterate_over_files(
                self.data_config, 
                self.data_files, 
                self.target_files, 
                self.read_source_func,
                self.read_source_kwargs,
            ),
            desc="Calculating data stats",
            total=len(self.data_files),
        ):            
            image_stats.update(sample, num_samples)

            # update the target statistics if target is available
            if target is not None:
                target_stats.update(target, num_samples)

            num_samples += 1

        if num_samples == 0:
            raise ValueError("No samples found in the dataset.")

        # Average the means and stds per sample
        image_means, image_stds = image_stats.finalize()

        if target is not None:
            target_means, target_stds = target_stats.finalize()

            return (
                Stats(means=image_means, stds=image_stds),
                Stats(means=target_means, stds=target_stds),
            )
        else:
            return Stats(means=image_means, stds=image_stds), Stats()
        
    def _calculate_min_and_max(self) -> tuple[Stats, Stats]:
        """Calculate channel-wise min and max of the dataset.

        Returns
        -------
        tuple[Stats, Stats]
            Data classes containing the image and target statistics.
        """
        num_samples = 0
        image_stats = RunningMinMaxStatistics()
        if self.target_files is not None:
            target_stats = RunningMinMaxStatistics()

        for sample, target, _ in tqdm(
            iterate_over_files(
                self.data_config,
                self.data_files,
                self.target_files,
                self.read_source_func,
                self.read_source_kwargs,
            ),
            desc="Calculating data stats",
            total=len(self.data_files),
        ):            
            image_stats.update(sample)

            # update the target statistics if target is available
            if target is not None:
                target_stats.update(target)

            num_samples += 1

        if num_samples == 0:
            raise ValueError("No samples found in the dataset.")

        # Get statistics
        image_mins, image_maxs = image_stats.mins, image_stats.maxs

        if target is not None:
            target_mins, target_maxs = target_stats.mins, target_stats.maxs
            return (
                Stats(mins=image_mins, maxs=image_maxs),
                Stats(mins=target_mins, maxs=target_maxs),
            )
        else:
            return Stats(mins=image_mins, maxs=image_maxs), Stats()
        
    def _set_mean_std_stats(self) -> None:
        """Set the mean and std image statistics."""
        if not self.data_config.image_means:
            self.image_stats, self.target_stats = self._calculate_mean_and_std()
            logger.info(
                f"Computed dataset mean: {self.image_stats.means},"
                f"std: {self.image_stats.stds}"
            )

            # update the mean in the config
            self.data_config.set_means_and_stds(
                image_means=self.image_stats.means,
                image_stds=self.image_stats.stds,
                target_means=self.target_stats.means,
                target_stds=self.target_stats.stds,
            )

        else:
            # if mean and std are provided in the config, use them
            self.image_stats, self.target_stats = (
                Stats(
                    means=self.data_config.image_means,
                    stds=self.data_config.image_stds
                ),
                Stats(
                    means=self.data_config.target_means,
                    stds=self.data_config.target_stds
                ),
            )
        
    def _set_min_max_stats(self) -> None:
        """Set the min and max image statistics."""
        if self.data_config.image_mins is None:
            self.image_stats, self.target_stats = self._calculate_min_and_max()
            logger.info(
                f"Computed dataset min: {self.image_stats.mins},"
                f"max: {self.image_stats.maxs}"
            )

            # update min and maxs in configuration
            self.data_config.set_mins_and_maxs(
                image_mins=self.image_stats.mins,
                image_maxs=self.image_stats.maxs,
                target_mins=self.target_stats.mins,
                target_maxs=self.target_stats.maxs
            )
        
        else:
            # if min and max are provided in the config, use them
            self.image_stats, self.target_stats = (
                Stats(
                    mins=self.data_config.image_mins,
                    maxs=self.data_config.image_maxs
                ),
                Stats(
                    mins=self.data_config.target_mins,
                    maxs=self.data_config.target_maxs
                ),
            )
    
    def _set_image_stats(self) -> None:
        """Set the image statistics."""
        if self.norm_type == "normalize":
            self._set_min_max_stats()
        elif self.norm_type == "standardize":
            self._set_mean_std_stats() 

    def __iter__(
        self,
    ) -> Generator[tuple[np.ndarray, ...], None, None]:
        """
        Iterate over data source and yield single patch.

        Yields
        ------
        np.ndarray
            Single patch.
        """
        assert (
            self.image_stats.means is not None and self.image_stats.stds is not None
        ), "Mean and std must be provided"

        # iterate over files
        for sample_input, sample_target, _ in iterate_over_files(
            self.data_config,
            self.data_files,
            self.target_files,
            self.read_source_func,
            self.read_source_kwargs,
        ):  
            patches = extract_patches_random(
                arr=sample_input,
                patch_size=self.data_config.patch_size,
                target=sample_target,
            )

            # iterate over patches
            # patches are tuples of (patch, target) if target is available
            # or (patch, None) only if no target is available
            # patch is of dimensions (C)ZYX
            for patch_data in patches:
                yield self.patch_transform(
                    patch=patch_data[0],
                    target=patch_data[1],
                )

    def get_data_statistics(self) -> tuple[list[float], list[float]]:
        """Return training data statistics.

        Returns
        -------
        tuple of list of floats
            Means and standard deviations across channels of the training data.
        """
        return self.image_stats.get_statistics()

    def get_number_of_files(self) -> int:
        """
        Return the number of files in the dataset.

        Returns
        -------
        int
            Number of files in the dataset.
        """
        return len(self.data_files)

    def split_dataset(
        self,
        percentage: float = 0.1,
        minimum_number: int = 5,
    ) -> PathIterableDataset:
        """Split up dataset in two.

        Splits the datest sing a percentage of the data (files) to extract, or the
        minimum number of the percentage is less than the minimum number.

        Parameters
        ----------
        percentage : float, optional
            Percentage of files to split up, by default 0.1.
        minimum_number : int, optional
            Minimum number of files to split up, by default 5.

        Returns
        -------
        IterableDataset
            Dataset containing the split data.

        Raises
        ------
        ValueError
            If the percentage is smaller than 0 or larger than 1.
        ValueError
            If the minimum number is smaller than 1 or larger than the number of files.
        """
        if percentage < 0 or percentage > 1:
            raise ValueError(f"Percentage must be between 0 and 1, got {percentage}.")

        if minimum_number < 1 or minimum_number > self.get_number_of_files():
            raise ValueError(
                f"Minimum number of files must be between 1 and "
                f"{self.get_number_of_files()} (number of files), got "
                f"{minimum_number}."
            )

        # compute number of files
        total_files = self.get_number_of_files()
        n_files = max(round(percentage * total_files), minimum_number)

        # get random indices
        indices = np.random.choice(total_files, n_files, replace=False)

        # extract files
        val_files = [self.data_files[i] for i in indices]

        # remove patches from self.patch
        data_files = []
        for i, file in enumerate(self.data_files):
            if i not in indices:
                data_files.append(file)
        self.data_files = data_files

        # same for targets
        if self.target_files is not None:
            val_target_files = [self.target_files[i] for i in indices]

            data_target_files = []
            for i, file in enumerate(self.target_files):
                if i not in indices:
                    data_target_files.append(file)
            self.target_files = data_target_files

        # clone the dataset
        dataset = copy.deepcopy(self)

        # reassign patches
        dataset.data_files = val_files

        # reassign targets
        if self.target_files is not None:
            dataset.target_files = val_target_files

        return dataset
