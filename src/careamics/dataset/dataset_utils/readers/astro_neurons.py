import os
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
import tifffile as tiff

from careamics.file_io.read.czi import _get_czi_shape, read_czi


class GroupType(Enum):
    """The groups of samples in the dataset.
    
    Each group corresponds to a specific treatment or condition.
    The list of strings associated to each group are the ones used in the filenames.
    """
    CONTROL = ["control", "untreated"]
    ARSENITE = ["arsenite", "NaAsO", "Ars"]
    THARPS = ["TG", "tharps"]


def _str_to_group_type(
    groups: Sequence[Union[Literal["control", "arsenite", "tharps"], GroupType]]
) -> Sequence[GroupType]:
    """Convert a list of strings to `GroupType` instances.
    
    Parameters
    ----------
    groups : Sequence[Union[Literal["control", "arsenite", "tharps"], GroupType]]
        The groups of samples to load.
    
    Returns
    -------
    Sequence[GroupType]
        The list of `GroupType` instances.
    """
    STR_TO_GROUP_TYPE = {
        "control": GroupType.CONTROL,
        "arsenite": GroupType.ARSENITE,
        "tharps": GroupType.THARPS,
    }
    if all([isinstance(g, str) for g in groups]):
        return [STR_TO_GROUP_TYPE[g] for g in groups]
    elif all([isinstance(g, GroupType) for g in groups]):
        return groups
    else:
        ValueError(
            "Invalid group type. All entries must be str or `GroupType` instances."
        )
        

def _sort_fnames(fnames: list[str]) -> list[str]:
    """Sort filenames to match 'raw' and 'unmixed' data in experiments.
    
    Sorting is done based on the ID of the experiment (e.g., VD-0428 < VD-0505)
    and the number of replicate of an image (number in the filename).
    
    Parameters
    ----------
    x : str
        The filename to sort.
    """
    def sorting_fn(x: str) -> int:
        dirname = os.path.basename(os.path.dirname(x))
        primary_key = int(dirname.split("_")[0][-3:])
        fname = os.path.basename(x)
        secondary_key = fname.split(".")[0].split("_")[3]
        try: # astrocytes
            secondary_key = int(secondary_key)
        except ValueError: # neurons
            secondary_key = int(secondary_key[-1])
        return primary_key, secondary_key
    
    return sorted(fnames, key=sorting_fn)
    

def get_fnames(
    data_path: Union[str, Path],
    dset_type: Literal["astrocytes", "neurons"],
    img_type: Literal["raw", "unmixed"],
    groups: Sequence[Union[Literal["control", "arsenite", "tharps"], GroupType]],
    dim: Literal["2D", "3D"],
) -> list[str]:
    """Get the filenames of the images to load.
    
    Parameters
    ----------
    data_path : Union[str, Path]
        The path to the data, specifically "/path/to/data/neurons_and_astrocytes".
    dset_type : Literal["astrocytes", "neurons"]
        The type of dataset to load.
    img_type : Literal["raw", "unmixed"]
        The type of image to load, i.e., either raw multispectral or unmixed stacks.
    groups : Sequence[Union[Literal["control", "arsenite", "tharps"], GroupType]]
        The groups of samples to load.
    dim : Literal["2D", "3D"]
        The dimensionality of the images to load.
    
    Returns
    -------
    list[str]
        The list of filenames to load, sorted by exp ID and img ID.
    """
    fnames = []
    dim_dir = "Z-stacks" if dim == "3D" else "slices"
    data_path = os.path.join(data_path, dset_type, dim_dir, img_type)
    subdirs = os.listdir(data_path)
    groups = _str_to_group_type(groups)
    for subdir in subdirs:
        subdir_path = os.path.join(data_path, subdir)
        for group in groups:
            allfiles = os.listdir(subdir_path)
            for alias in group.value:
                fnames += [
                    os.path.join(subdir_path, f) for f in allfiles if alias in f
                ]
    return _sort_fnames(fnames)


def split_train_test_fnames(
    fnames: list[str],
    test_percent: float = 0.1,
    stratify: bool = False,
    deterministic: bool = False,
) -> list[list[str], list[str]]:
    """Split the list of filenames into training and testing sets.
    
    Parameters
    ----------
    fnames : list[str]
        The list of filenames to split.
    test_percent : float
        The percentage of data to use for testing.
    stratify : bool
        Whether to stratify the split by group. Default is False.
    deterministic : bool
        Whether to use a fixed seed for reproducibility. Default is False.
        
    Returns
    -------
    list[list[str], list[str]]
        The training and testing sets.
    """
    n_train = int(len(fnames) * (1 - test_percent))
    if stratify:
        raise NotImplementedError("Stratified split not implemented yet.")
    if deterministic:
        return fnames[:n_train], fnames[n_train:]
    else:
        train_idxs = np.random.choice(len(fnames), n_train, replace=False)
        test_idxs = np.setdiff1d(np.arange(len(fnames)), train_idxs)
        return [fnames[i] for i in train_idxs], [fnames[i] for i in test_idxs]


def get_max_z_size(fnames: list[str]) -> int:
    """Get the maximum Z size of the images in `fnames`.
    
    Parameters
    ----------
    fnames : list[str]
        The list of filenames to load.

    Returns
    -------
    int
        The maximum Z size.
    """
    return np.max([_get_czi_shape(fname)[1] for fname in fnames])


def load_3D_img(fpath: Union[str, Path], max_z: int) -> NDArray:
    """Load a 3D Z-stack image from a file and pad Z dimension.
    
    Parameters
    ----------
    fpath : Union[str, Path]
        The path to the image file.
    
    Returns
    -------
    NDArray
        The loaded image.
    """
    img = read_czi(fpath).squeeze()
    z = img.shape[1]
    pad = max_z - z
    pad_up = pad // 2
    pad_down = pad - pad_up
    return np.pad(img, ((0, 0), (pad_down, pad_up), (0, 0), (0, 0)), mode="constant")


def _load_3D_data(fnames: list[str]) -> NDArray:
    """Load 3D Z-stack images.
    
    The problem here is that the images are not necessarily the same size along Z.
    Hence, we first need to load all images to get the maximum Z size, and then pad.
    
    Parameters
    ----------
    fnames : list[str]
        The list of filenames to load.
        
    Returns
    -------
    NDArray
        The loaded data. Shape is (N, C, Z, Y, X).
    """
    # --- get the maximum Z size
    max_z = get_max_z_size(fnames)
    
    # --- load and pad images
    data = []
    for fname in tqdm(fnames, desc="Loading images"):
        data.append(load_3D_img(fname, max_z))
    return np.array(data)


def _load_2D_data(fnames: list[str]) -> NDArray:
    """Load 2D slice images.
    
    Parameters
    ----------
    fnames : list[str]
        The list of filenames to load.
        
    Returns
    -------
    NDArray
        The loaded data. Shape is (N, C, Y, X).
    """
    data = []
    for fname in tqdm(fnames, desc="Loading images"):
        img = tiff.imread(fname)
        data.append(img)
    return np.array(data)
  

def load_astro_neuron_data(
    data_path: Union[str, Path],
    dset_type: Literal["astrocytes", "neurons"], # TODO: change name
    img_type: Literal["raw", "unmixed"], # TODO: change name
    groups: Sequence[Union[Literal["control", "arsenite", "tharps"], GroupType]],
    dim: Literal["2D", "3D"] = "2D",
    split: Optional[Literal["train", "test"]] = None,
    test_percent: float = 0.1,
    deterministic_split: bool = True,
    only_first_n: Optional[int] = None, # TODO: tmp, this won't work for multi groups
) -> NDArray:
    """Load data from neurons & astrocytes dataset.
    
    Parameters
    ----------
    data_path : Union[str, Path]
        The path to the data, specifically "/path/to/data/neurons_and_astrocytes".
    dset_type : Literal["astrocytes", "neurons"]
        The type of dataset to load.
    img_type : Literal["raw", "unmixed"]
        The type of image to load, i.e., either raw multispectral or unmixed stacks.
    groups : Sequence[Union[Literal["control", "arsenite", "tharps"], GroupType]]
        The groups of samples to load.
    dim : Literal["2D", "3D"]
        Whether to load 3D Z-stacks or 2D slices.
    split : Optional[Literal["train", "test"]]
        Whether to load the training or testing set. If None all data are taken.
        Default is None.
    test_percent : float
        The percentage of data to use for testing. Default is 0.1.
    deterministic_split : bool
        Whether to split train and test deterministically. Default is `True`.
    only_first_n : Optional[int]
        Load only the first `n` filenames. Default is None.
        
    Returns
    -------
    NDArray
        The loaded data. Shape is (N, C, Z, Y, X) if `get_2D` is False, otherwise
        (N, C, Y, X).
    """
    groups = _str_to_group_type(groups)
    fnames = get_fnames(
        data_path=data_path, 
        dset_type=dset_type, 
        dim=dim,
        img_type=img_type, 
        groups=groups
    )
    if split is not None:
        train_fnames, test_fnames = split_train_test_fnames(
            fnames, test_percent=test_percent, deterministic=deterministic_split
        )
        fnames = train_fnames if split == "train" else test_fnames
    fnames = fnames[:only_first_n] if only_first_n is not None else fnames
    print(
        f"Dataset: {dset_type} -- {img_type} -- {[g.name for g in groups]} -- {dim}"
        f" -- {split}"
    )
    print(f"Found {len(fnames)} images.")
    if dim == "3D":
        data = _load_3D_data(fnames)
    else:
        data = _load_2D_data(fnames)
    return data