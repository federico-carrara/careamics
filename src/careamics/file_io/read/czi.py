from pathlib import Path
from typing import Union

import czifile as czi
from numpy.typing import NDArray


def read_czi(
    path: Union[str, Path], *args: list, **kwargs: dict
) -> Union[NDArray, tuple[NDArray, dict]]:
    """Load a CZI file and return the image and optionally the metadata.
    
    Parameters
    ----------
    path : Union[str, Path]
        The path to the CZI file.
    
    Returns
    -------
    Union[NDArray, tuple[NDArray, dict]]
        The image and optionally the metadata.
    """
    with czi.CziFile(path) as f:
        img = f.asarray()
        return img.squeeze()