import numpy as np
import pytest
from skimage.metrics import peak_signal_noise_ratio

from careamics.utils.metrics import (
    _zero_mean,
    psnr,
    scale_invariant_psnr,
)


@pytest.mark.parametrize(
    "x",
    [
        np.array([1, 2, 3, 4, 5]),
        np.array([[1, 2, 3], [4, 5, 6]]),
    ],
)
def test_zero_mean(x: np.ndarray):
    assert np.allclose(_zero_mean(x), x - np.mean(x))


# TODO: with 2 identical arrays, shouldn't the result be `inf`?
@pytest.mark.parametrize(
    "gt, pred, result",
    [
        (np.array([1, 2, 3, 4, 5, 6]), np.array([1, 2, 3, 4, 5, 6]), 332.22),
        (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2, 3], [4, 5, 6]]), 332.22),
    ],
)
def test_scale_invariant_psnr(gt: np.ndarray, pred: np.ndarray, result: float):
    assert scale_invariant_psnr(gt, pred) == pytest.approx(result, rel=5e-3)


@pytest.mark.parametrize("data_type", [np.uint8, np.uint16, np.float32, np.int64])
@pytest.mark.parametrize("range_type", ["min-max", "skimage_default"])
def test_psnr_internal(data_type: np.dtype, range_type: str):  
    gt = np.random.randint(0, 255, size=(16, 16)).astype(data_type)
    pred = np.random.randint(0, 255, size=(16, 16)).astype(data_type)
    # compute expected PSNR
    mse = ((gt - pred) ** 2).mean()
    data_range = gt.max() - gt.min()
    exp_psnr = 10 * np.log10(data_range ** 2 / mse)
    assert psnr(gt, pred, None) == pytest.approx(exp_psnr, rel=5e-3)
    assert psnr(gt, pred, data_range) == pytest.approx(exp_psnr, rel=5e-3)

    
@pytest.mark.skip("Not working as expected.")
@pytest.mark.parametrize("data_type", [np.uint8, np.uint16, np.float32, np.int64])
@pytest.mark.parametrize("range_type", ["min-max", "skimage_default"])
def test_psnr_skimage(data_type: np.dtype, range_type: str):  
    DTYPE2RANGE = {
        t: np.iinfo(t).max - np.iinfo(t).min
        for t in [np.uint8, np.uint16, np.int64]
    }
    DTYPE2RANGE[np.float32] = 2
    gt = np.random.randint(0, 255, size=(16, 16)).astype(data_type)
    pred = np.random.randint(0, 255, size=(16, 16)).astype(data_type)
    # compute expected PSNR
    mse = ((gt - pred) ** 2).mean()
    if range_type == "min-max":
        data_range = gt.max() - gt.min()
    elif range_type == "skimage_default":
        data_range = DTYPE2RANGE[data_type]
    exp_psnr = 10 * np.log10(data_range ** 2 / mse)
    assert psnr(gt, pred, data_range, peak_signal_noise_ratio) == pytest.approx(exp_psnr, rel=5e-3)
    


@pytest.mark.skip(reason="Not implemented")
def test_fix_range():
    pass