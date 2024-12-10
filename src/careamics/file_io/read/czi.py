from pathlib import Path
from typing import Union

import czifile as czi
import numpy as np
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
    
    
def _get_czi_shape(path: Union[str, Path]) -> tuple[int]:
    """Get the shape of a CZI image.
    
    Parameters
    ----------
    path : Union[str, Path]
        The path to the CZI file.
    
    Returns
    -------
    tuple[int]
        The shape of the image without singleton dims. We expect the shape to be:
        - (Y, X) and (Z, Y, X) for 2D and 3D images, respectively.
        - (C, Y, X) and (C, Z, Y, X) for 2D and 3D multispectral images, respectively.
    """
    with czi.CziFile(path) as f:
        shp = np.asarray(f.shape)
        return shp[shp != 1]
