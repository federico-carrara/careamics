import os
from enum import Enum
from pathlib import Path
from typing import Literal, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
import tifffile as tiff

from careamics.dataset.dataset_utils.readers.utils import load_czi

# NOTE: for the moment we solely use on raw data

class GroupType(Enum):
    """The groups of samples in the dataset.
    
    Each group corresponds to a specific treatment or condition.
    The list of strings associated to each group are the ones used in the filenames.
    """
    CONTROL = ["control", "untreated"]
    ARSENITE = ["arsenite", "NaAsO", "Ars"]
    THARPS = ["TG", "tharps"]
    

def _load_img(fpath: Union[str, Path]) -> NDArray:
    if fpath.endswith(".czi"):
        img = load_czi(fpath).squeeze()
    elif fpath.endswith(".tif") or fpath.endswith(".tiff"):
        img = tiff.imread(fpath)
    else:
        raise ValueError(
            f"Unsupported file format: {fpath}. Supported formats are .czi and .tif."
        )
    return img
    

def _get_fnames(
    data_path: Union[str, Path],
    dset_type: Literal["astrocytes", "neurons"],
    img_type: Literal["raw", "unmixed"],
    groups: Sequence[GroupType],
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
    groups : Sequence[AstroGroupType]
        The groups of samples to load.
    dim : Literal["2D", "3D"]
        The dimensionality of the images to load.
    
    Returns
    -------
    list[str]
        The list of filenames to load.
    """
    assert img_type == "raw", "Only raw data is supported for now."
    fnames = []
    dim_dir = "Z-stacks" if dim == "3D" else "slices"
    data_path = os.path.join(data_path, dset_type, dim_dir, img_type)
    subdirs = os.listdir(data_path)
    for subdir in subdirs:
        subdir_path = os.path.join(data_path, subdir)
        for group in groups:
            allfiles = os.listdir(subdir_path)
            for alias in group.value:
                fnames += [
                    os.path.join(subdir_path, f) for f in allfiles if alias in f
                ]
    return fnames


def _get_mid_slice(img: NDArray) -> NDArray:
    """Get the middle Z-slice of a 3D stack.
    
    Parameters
    ----------
    img : NDArray
        The 3D stack. Shape is (C, Z, Y, X).
        
    Returns
    -------
    NDArray
        The middle Z-slice. Shape is (C, Y, X).
    """
    z_dim = img.shape[1]
    mid_z = z_dim // 2
    return img[:, mid_z, :, :]


def load_astro_neuron_data(
    data_path: Union[str, Path],
    dset_type: Literal["astrocytes", "neurons"],
    img_type: Literal["raw", "unmixed"],
    groups: Sequence[GroupType],
    dim: Literal["2D", "3D"] = "2D",
) -> NDArray:
    """Load data from neurons & astrocytes dataset.
    
    Data is naturally in the shape of multispectral 3D stacks, i.e., (C, Z, Y, X).
    By setting `get_2D` to True, the function will return the middle Z-slice of stacks.
    
    Parameters
    ----------
    data_path : Union[str, Path]
        The path to the data, specifically "/path/to/data/neurons_and_astrocytes".
    dset_type : Literal["astrocytes", "neurons"]
        The type of dataset to load.
    img_type : Literal["raw", "unmixed"]
        The type of image to load, i.e., either raw multispectral or unmixed stacks.
    groups : Sequence[GroupType]
        The groups of samples to load.
    dim : Literal["2D", "3D"]
        Whether to load 3D Z-stacks or 2D slices.
        
    Returns
    -------
    NDArray
        The loaded data. Shape is (N, C, Z, Y, X) if `get_2D` is False, otherwise
        (N, C, Y, X).
    """
    fnames = _get_fnames(
        data_path=data_path, 
        dset_type=dset_type, 
        dim=dim,
        img_type=img_type, 
        groups=groups
    )
    print(f"Dataset: {dset_type} -- {img_type} -- {[g.name for g in groups]} -- {dim}")
    print(f"Found {len(fnames)} images.")
    data = []
    for fname in tqdm(fnames, desc="Loading images"):
        img = _load_img(fname)
        data.append(img) # TODO: use generator for memory efficiency (yield)
    return np.array(data)