from pathlib import Path
from typing import Union

import czifile as czi
from numpy.typing import NDArray


def read_czi(
    path: Union[str, Path], load_metadata: bool =  False
) -> Union[NDArray, tuple[NDArray, dict]]:
    """Load a CZI file and return the image and optionally the metadata.
    
    Parameters
    ----------
    path : Union[str, Path]
        The path to the CZI file.
    load_metadata : bool, optional
        Whether to load the metadata, by default False.
    
    Returns
    -------
    Union[NDArray, tuple[NDArray, dict]]
        The image and optionally the metadata.
    """
    with czi.CziFile(path) as f:
        img = f.asarray()
        if load_metadata:
            metadata = f.metadata()
            return img, metadata
        else:
            return img