"""Set of functions to visualize images."""

from typing import Optional

from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray


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