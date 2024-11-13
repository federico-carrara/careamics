"""Set of functions to visualize images."""

from typing import Optional

from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray
import torch
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from careamics.utils.metrics import scale_invariant_psnr
        
        
def _get_animation(
    imgs: list[np.ndarray], 
    axis: int,
    titles: Optional[list[str]] = None
) -> FuncAnimation:
    """Get the matplotlib animation for multiple images.

    Parameters
    ----------
    imgs : Union[np.ndarray, List[np.ndarray]]
        A 3D image array or a list of 3D image arrays to plot.
    axis : int
        Axis to scroll through, common to all images.
    titles: Optional[list[str]]
        Titles for each image. Default is `None`.

    Returns
    -------
    FuncAnimation
        Matplotlib animation.
    """
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
    
    assert all(img.ndim == 3 for img in imgs), "Each image must be 3D."
    assert all(img.shape == imgs[0].shape for img in imgs), "All images must have the same shape."
    assert axis < imgs[0].ndim, "Scroll axis out of range."
    
    plt.rcParams["animation.html"] = "jshtml"

    num_images = len(imgs)
    fig, axs = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    
    # Ensure axs is iterable when there's only one subplot
    if num_images == 1:
        axs = [axs]

    slices = [slice(None)] * imgs[0].ndim
    slices[axis] = 0
    
    ims = []
    for img, ax in zip(imgs, axs):
        ax.set_title(titles.pop(0) if titles else "")
        im = ax.imshow(img[tuple(slices)])
        ims.append(im)
        
    # Update function for all images
    def update(frame: int):
        slices[axis] = frame
        for im, img in zip(ims, imgs):
            im.set_array(img[tuple(slices)])
        return ims

    return FuncAnimation(fig, update, frames=imgs[0].shape[axis], interval=200)


def view3D(
    imgs: list[np.ndarray],
    titles: Optional[list[str]] = None,
    axis: Optional[int] = 0, 
    jupyter: Optional[bool] = True,
    save_path: Optional[str] = None
) -> Optional[HTML]:
    """View one or multiple 3D images.

    Parameters
    ----------
    imgs : List[np.ndarray]
        List of 3D images to plot. Each image must have the same shape.
    axis : Optional[int]
        Axis to scroll through. Default is 0.
    jupyter : Optional[bool]
        Whether to display in a Jupyter Notebook. Default is True.
    save_path : Optional[str]
        Path to save the animation. Default is None.

    Returns
    -------
    Optional[HTML]
        HTML animation if in Jupyter Notebook, else `None`.
    """
    anim = _get_animation(imgs, axis, titles)
    
    # Save
    if save_path:
        anim.save(save_path, writer='pillow')
    
    # Render
    if jupyter:
        return HTML(anim.to_jshtml())
    else:    
        plt.show()


def intensity_histograms(
    imgs: torch.Tensor,
    max_y: float = None,
    max_x: float = None
) -> None:
    """Plot intensity histograms of a multichannel image.
    
    Parameters
    ----------
    imgs : torch.Tensor
        Multichannel image. Shape is (C, [Z], Y, X).
    max_y : float
        Maximum y-axis value.
    max_x : float
        Maximum x-axis value.
        
    Returns
    -------
    None
    """
    n = imgs.shape[0]
    fig, ax = plt.subplots(1, n, figsize=(5*n, 5))
    fig.suptitle(f"Intensity distributions")
    for i in range(n):
        img = imgs[i]
        img = img.flatten()
        qtiles = np.quantile(img, (0.5, 0.75, 0.95, 0.99))
        mean = img.mean()
        std = img.std()
        ax[i].set_title(f"Intensity distr: FP#{i+1}")
        ax[i].hist(img, bins=20, color="gray")
        for qt in qtiles:
            ax[i].axvline(qt, color='r', linestyle='--', linewidth=1)
        ax[i].set_ylim(0, max_y)
        ax[i].set_xlim(0, max_x)
        print(f"FP#{i+1}: mean: {mean:.2f}, std: {std:.2f}, 50%: {qtiles[0]:.2f}, 75%: {qtiles[1]:.2f}, 95%: {qtiles[2]:.2f}, 99%: {qtiles[3]:.2f}")

    plt.show()


def _add_colorbar(img: torch.Tensor, fig, ax):
    """Add colorbar to a `matplotlib` image."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)


def plot_splitting_results(
    preds: torch.Tensor,
    gts: torch.Tensor,
    preds_std: Optional[torch.Tensor] = None,
    idx: Optional[int] = None,
) -> None:
    """Plot a predicted image with the associated GT.
    
    Parameters
    ----------
    preds : torch.Tensor
        The predicted image.
    gts : torch.Tensor
        The ground truth image.
    preds_std : Optional[torch.Tensor]
        The predicted std deviation. If `None`, it won't be plotted. Default is `None`.
    idx : Optional[int], optional
        The index of the image to plot, by default None.
    """
    N, F = preds.shape[0], preds.shape[1]
    
    if idx is None:
        idx = np.random.randint(0, N - 1)
        
    ncols = 4 if preds_std else 3
    fig, axes = plt.subplots(F, ncols, figsize=(6 * ncols, 5 * F))
    for i in range(F):
        # GT
        axes[i, 0].set_title(f"FP {i+1} - GT")
        im_gt = axes[i, 0].imshow(gts[idx, i])
        _add_colorbar(im_gt, fig, axes[i, 0])
        # Pred
        axes[i, 1].set_title(f"FP {i+1} - Pred")
        im_pred = axes[i, 1].imshow(preds[idx, i])
        _add_colorbar(im_pred, fig, axes[i, 1])
        # MAE
        pred, gt = preds[idx, i], gts[idx, i]
        norm_pred = (pred - pred.min()) / (pred.max() - pred.min())
        norm_gt = (gt - gt.min()) / (gt.max() - gt.min())
        mae = np.abs(norm_pred - norm_gt)
        axes[i, 2].set_title(f"FP {i+1} - MAE")
        im_mae = axes[i, 2].imshow(mae, cmap='RdPu')
        _add_colorbar(im_mae, fig, axes[i, 2])
        psnr = scale_invariant_psnr(pred, gt)
        axes[i, 2].text(
            0.66, 0.1, f'PSNR: {psnr:.2f}', 
            transform=axes[i, 2].transAxes,
            fontsize=12, 
            verticalalignment='center', 
            bbox=dict(facecolor='white', alpha=0.5)
        )
        # Pred Std
        if preds_std:
            axes[i, 3].set_title(f"FP {i+1} - Pred Std")
            im_std = axes[i, 3].imshow(preds_std[idx, i])
            _add_colorbar(im_std, fig, axes[i, 3])
        