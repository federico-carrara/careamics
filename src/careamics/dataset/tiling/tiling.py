from numpy.typing import NDArray

from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from tqdm import tqdm

from careamics.config.tile_information import TileInformation
from careamics.dataset.dataset_utils import reshape_array
from careamics.dataset.dataset_utils.synthetic_noise import SyntheticNoise
from careamics.dataset.tiling import extract_tiles
from careamics.utils.logging import get_logger

logger = get_logger(__name__)


def prepare_tiles(
    fpaths: Sequence[Path],
    axes: str,
    tile_size: Sequence[int],
    tile_overlap: Sequence[int],
    read_source_func: Callable,
    read_source_kwargs: Optional[dict[str, Any]],
    synthetic_noise: Optional[SyntheticNoise] = None,
) -> list[tuple[NDArray, TileInformation]]:
    """Prepare tiles from a sequence of file paths.
    
    Parameters
    ----------
    fpaths : Sequence[Path]
        Sequence of file paths.
    axes : str
        Axes of the input data.
    tile_size : Sequence[int]
        Size of the tiles.
    tile_overlap : Sequence[int]
        Overlap between tiles.
    read_source_func : Callable
        Function to read the source data.
    read_source_kwargs : Optional[dict[str, Any]]
        Keyword arguments for the read_source_func.
    synthetic_noise : Optional[SyntheticNoise]
        Synthetic noise object to apply to the data. Default is None.
    
    Returns
    -------
    list[tuple[np.ndarray, TileInformation]]
        List of tuples containing the tile and its information.
    """
    if read_source_kwargs is None:
        read_source_kwargs = {}
    
    num_samples = -1
    tile_list = []
    for filename in tqdm(fpaths, desc="Reading files"):
        try:
            sample: NDArray = read_source_func(filename, **read_source_kwargs)
            num_samples += 1
            
            # apply synthetic noise
            if synthetic_noise is not None:
                sample = synthetic_noise(sample, axes=axes)

            # reshape array
            sample = reshape_array(sample, axes)

            # generate tiles, return a generator
            tile_generator = extract_tiles(
                arr=sample,
                tile_size=tile_size,
                overlaps=tile_overlap,
                file_id=num_samples,
            )

            # convert generator to list and add tile_list
            tile_list.extend(list(tile_generator))
        except Exception as e:
            # emit warning and continue
            logger.error(f"Failed to read {filename}: {e}")

    # raise error if no valid samples found
    if num_samples == 0:
        raise ValueError(f"No valid samples found in the input data: {fpaths}.")

    logger.info(f"Extracted {len (tile_list)} tiles from input array.")
    return tile_list


def prepare_tiles_array(
    data: NDArray,
    axes: str,
    tile_size: Sequence[int],
    tile_overlap: Sequence[int],
    synthetic_noise: Optional[SyntheticNoise] = None,
) -> list[tuple[NDArray, TileInformation]]:
    """Prepare tiles from an array of images.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array.
    axes : str
        Axes of the input data.
    tile_size : Sequence[int]
        Size of the tiles.
    tile_overlap : Sequence[int]
        Overlap between tiles.
    synthetic_noise : Optional[SyntheticNoise]
        Synthetic noise object to apply to the data. Default is None.
        
    Returns
    -------
    list[tuple[np.ndarray, TileInformation]]
        List of tuples containing the tile and its information.
    """
    # apply synthetic noise
    if synthetic_noise is not None:
        data = synthetic_noise(data, axes=axes)
    
    # reshape array
    reshaped_sample = reshape_array(data, axes)

    # generate patches, which returns a generator
    tile_generator = extract_tiles(
        arr=reshaped_sample,
        tile_size=tile_size,
        overlaps=tile_overlap,
    )
    return list(tile_generator)