"""Set of functions to visualize images."""

from typing import Optional

from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray
import torch
import numpy as np

def _get_animation(img: NDArray, axis: int) -> FuncAnimation:
    """Get the matplotlib animation.

    Parameters
    ----------
    img : NDArray
        Image to plot. Shape must be 3D.
    axis : int
        Axis to scroll through.

    Returns
    -------
    FuncAnimation
        Matplotlib animation.
    """
    assert img.ndim == 3, "Image must be 3D."
    assert axis < img.ndim, "Scroll axis out of range."
    
    plt.rcParams["animation.html"] = "jshtml"

    fig, ax = plt.subplots()
    
    slices = [slice(None)] * img.ndim
    slices[axis] = 0
    im = ax.imshow(img[tuple(slices)])
        
    # update function that defines the matplotlib animation
    def update(frame: int):
        slices[axis] = frame
        im.set_array(img[tuple(slices)])
        return [im]

    return FuncAnimation(
        fig, update, frames=img.shape[axis], interval=200
    )


def view3D(img: NDArray, axis: int, jupyter: bool) -> Optional[HTML]:
    """View 3D image.

    Parameters
    ----------
    img : NDArray
        Image to plot. Shape must be 3D.
    axis : int
        Axis to scroll through.
    jupyter : bool
        Whether to display in a Jupyter Notebook.
        
    Returns
    -------
    Optional[HTML]
        HTML animation if in Jupyter Notebook, else `None`.
    """
    anim = _get_animation(img, axis)
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