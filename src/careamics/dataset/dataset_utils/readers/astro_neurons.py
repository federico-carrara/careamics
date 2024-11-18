import os
from enum import Enum
from pathlib import Path
from typing import Literal, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

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
    

def _get_fnames(
    data_path: Union[str, Path],
    dset_type: Literal["astrocytes", "neurons"],
    img_type: Literal["raw", "unmixed"],
    groups: Sequence[GroupType],
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
    
    Returns
    -------
    list[str]
        The list of filenames to load.
    """
    assert img_type == "raw", "Only raw data is supported for now."
    fnames = []
    data_path = os.path.join(data_path, dset_type, "Z-stacks", img_type)
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
    get_2D: bool
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
    groups : Sequence[NeuronGroupType]
        The groups of samples to load.
    get_2D : bool
        Whether to return the middle Z-slice of stacks.
        
    Returns
    -------
    NDArray
        The loaded data. Shape is (N, C, Z, Y, X) if `get_2D` is False, otherwise
        (N, C, Y, X).
    """
    fnames = _get_fnames(data_path, dset_type, img_type, groups)
    print(f"Loading {len(fnames)} images...")
    print(f"File names: {fnames}")
    data = []
    for fname in tqdm(fnames, desc="Loading images"):
        img = load_czi(fname).squeeze()
        if get_2D:
            img = _get_mid_slice(img)
        data.append(img)
    return np.array(data)


if __name__ == "__main__":
    DATA_PATH = Path("/group/jug/federico/data/neurons_and_astrocytes")
    imgs = load_astro_neuron_data(
        data_path=DATA_PATH,
        dset_type="neurons",
        img_type="unmixed",
        groups=[GroupType.CONTROL, GroupType.ARSENITE],
        get_2D=True
    )