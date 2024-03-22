import pytest

from torch import from_numpy

from careamics.dataset.patching.tiled_patching import extract_tiles
from careamics.prediction.prediction_utils import stitch_prediction


@pytest.mark.parametrize(
    "input_shape, tile_size, overlaps",
    [
        ((1, 1, 8, 8), (4, 4), (2, 2)),
        ((1, 1, 8, 8), (4, 4), (2, 2)),
        ((1, 1, 7, 9), (4, 4), (2, 2)),
        ((1, 1, 9, 7, 8), (4, 4, 4), (2, 2, 2)),
        ((1, 1, 321, 481), (256, 256), (48, 48)),
    ],
)
def test_stitch_prediction(ordered_array, input_shape, tile_size, overlaps):
    """Test calculating stitching coordinates.
    """
    arr = ordered_array(input_shape, dtype=int)
    tiles = []
    stitching_data = []

    # extract tiles
    tiling_outputs = extract_tiles(arr, tile_size, overlaps)

    # Assemble all tiles as it is done during the prediction stage
    for tile_data in tiling_outputs:
        tile, _, input_shape, overlap_crop_coords, stitch_coords = tile_data

        tiles.append(
            from_numpy(tile) # need to convert to torch.Tensor
        )

        stitching_data.append(
            (
                input_shape,
                overlap_crop_coords,
                stitch_coords,
            )
        )
        
    # compute stitching coordinates
    result = stitch_prediction(tiles, stitching_data)

    assert (result == arr).all()
