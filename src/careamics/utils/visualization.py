"""Set of functions to visualize images."""

from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray


def view3D(img: NDArray, axis: int) -> FuncAnimation:
    """Plot 3D image in a Jupyter Notebook.

    Parameters
    ----------
    img : NDArray
        Image to plot.
    axis : int
        Axis to scroll through.
    """
    assert axis < img.ndim, "Scroll axis out of range."
    
    plt.rcParams["animation.html"] = "jshtml"

    fig, ax = plt.subplots()
    
    slices = [slice(None)] * img.ndim
    slices[axis] = 0
    im = ax.imshow(img[tuple(slices)])
        
    # update function that defines the matplotlib animation
    def update(frame: int):
        im.set_array(img[frame])
        return [im]

    return FuncAnimation(
        fig, update, frames=img.shape[axis], interval=100
    )