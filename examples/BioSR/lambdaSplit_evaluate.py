import argparse
import glob
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import numpy as np
import torch
import tifffile as tiff
from torch.utils.data import DataLoader

from careamics.config import InferenceConfig
from careamics.config import (
    VAEAlgorithmConfig,
    DataConfig,
)
from careamics.dataset import InMemoryTiledPredDataset
from careamics.dataset.tiling import collate_tiles
from careamics.lightning import VAEModule
from careamics.prediction_utils import stitch_prediction
from careamics.utils.io_utils import load_config, load_model_checkpoint
from careamics.utils.eval_utils import coarsen_img, get_tiled_predictions
from careamics.utils.metrics import avg_range_inv_psnr
from careamics.utils.visualization import plot_splitting_results, view3D


# DATA_ROOT = "/group/jug/federico/microsim/BIOSR_spectral_data/2411/v0/imgs/"
# OUT_ROOT = "/group/jug/federico/lambdasplit_training/"
# ckpt_dir = os.path.join(OUT_ROOT, "2411/lambdasplit_no_LC/0")
# assert os.path.exists(ckpt_dir)

# Eval Parameters
mmse_count: int = 2
"""The number of predictions to average for MMSE evaluation."""
tile_size: list[int] = [64, 64]
"""The size of the portion of image we retain from inner padding/tiling."""
tile_overlap: list[int] = [32, 32]
"""The actual patch size. If not specified data.image_size."""
psnr_type: Literal['simple', 'range_invariant'] = 'range_invariant'
"""The type of PSNR to compute."""
which_ckpt: Literal['best', 'last'] = 'best'
"""Which checkpoint to use for evaluation."""
gt_type: Literal["optical", "digital"] = "optical"
"""The type of ground truth to use for evaluation. Note that optical needs to be downscaled."""

# Optional other params
batch_size: int = 32
"""The batch size for training."""
num_workers: int = 4
"""The number of workers to use for data loading."""


def read_multifile_tiff(
    path_to_dir: Union[str, Path], 
    sort_fn: Optional[Callable] = None
) -> np.ndarray:
    """Read a directory of tiff files and concatenates in a numpy array.
    
    Parameters
    ----------
    path_to_dir : Union[str, Path]
        Path to the directory containing tiff files.
    sort_fn : Optional[Callable], optional
        Function to sort the files before reading them, by default None.
    """
    files = glob.glob(os.path.join(path_to_dir, "*.tif"))
    assert len(files) > 0, f"No files found in {path_to_dir}"
    print(f"Reading {len(files)} files from {path_to_dir}...")
    if sort_fn:
        files = sorted(files, key=sort_fn)
    arrs = [np.array(tiff.imread(f)) for f in files]
    return np.stack(arrs)

def sorting_key(x: str) -> int:
    return int(x.split("/")[-1].split(".")[0].split("_")[-1])

def fix_seeds():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True


def evaluate(ckpt_dir: str, data_dir: str) -> None:
    
    fix_seeds()
    
    # Load/define configs
    if os.path.isdir(ckpt_dir):
        algo_config = VAEAlgorithmConfig(**load_config(ckpt_dir, "algorithm"))
        data_config = DataConfig(**load_config(ckpt_dir, "data"))
    
    pred_config = InferenceConfig(
        data_type="array",
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        batch_size=batch_size,
        axes="SCYX",
        image_means=data_config.image_means,
        image_stds=data_config.image_stds,
        tta_transforms=False
    )
    
    # Load data
    path_to_data = os.path.join(data_dir, "digital")
    input_data = read_multifile_tiff(path_to_data, sorting_key)
    val_dset = InMemoryTiledPredDataset(
        prediction_config=pred_config,
        inputs=input_data,
    ) # TODO: eval on subset of data
    val_dloader = DataLoader(
        val_dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_tiles
    )
    
    # Load GT data
    path_to_gt_data = os.path.join(data_dir, f"{gt_type}_pf/")
    gt_data = read_multifile_tiff(path_to_gt_data, sorting_key)
    # shape: (100, 3, 251, 251) (digital) or (100, 3, 1004, 1004) (optical)
    if gt_type == "optical":
        gt_data = np.concatenate([
            coarsen_img(img, 4)[None, ...] for img in gt_data
        ])
    
    # Load model & checkpoints
    light_model = VAEModule(algorithm_config=algo_config)
    checkpoint = load_model_checkpoint(ckpt_dir, which_ckpt)
    light_model.load_state_dict(checkpoint['state_dict'], strict=False)
    light_model.eval()
    light_model.cuda()
    print('Loading weights from epoch', checkpoint['epoch'])
    
    # Get tiled predictions
    pred_patches, pred_stds, rec_patches, tiles_info = get_tiled_predictions(
        light_model, val_dloader, mmse_count
    )
    
    # Stitch predictions
    pred_imgs = np.concatenate(stitch_prediction(pred_patches, tiles_info))
    pred_std_imgs = np.concatenate(stitch_prediction(pred_stds, tiles_info))
    rec_imgs = np.concatenate(stitch_prediction(rec_patches, tiles_info))
    
    # Compute metrics
    print("Model's outputs")
    print("---------------")
    psnr_arr = []
    for i in range(pred_imgs.shape[1]):
        psnr_arr.append(avg_range_inv_psnr(pred_imgs[:, i], gt_data[:, i]))
        print(f"Range-Invariant PSNR for FP#{i+1}: {psnr_arr[-1]:.2f}")
    print(f"Avg Range-Invariant PSNR: {np.mean(psnr_arr):.2f}")
    print(f"Combined Range-Invariant PSNR: {avg_range_inv_psnr(pred_imgs, gt_data):.2f}")
    
    # Plot some results
    plot_splitting_results(pred_imgs, pred_std_imgs, gt_data, idx=0)
    view3D(
        imgs=[input_data[0], rec_imgs[0]], 
        axis=0, 
        titles=["Input", "Reconstruction"], 
        jupyter=True,
        save_path=os.path.join(ckpt_dir, "reconstruction_example.gif")
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str)
    parser.add_argument("--data-dir", type=str)
    args = parser.parse_args()
    evaluate(args.ckpt_dir, args.data_dir)