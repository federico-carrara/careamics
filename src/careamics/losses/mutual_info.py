"""Implementation of differentiable histogram functions for the computation of mutual information.

Inspired by:
- https://matthew-brett.github.io/teaching/mutual_information.html
"""

from typing import Literal, Optional

import numpy as np
import torch
from torch import Tensor


# TODO: refactoring: code is highly duplicated and redundant. Create a class?
def _gaussian_kernel_binning(x: Tensor, centers: Tensor,sigma: float) -> Tensor:
    """Bin the input tensor using Gaussian kernels centered on the bin centers.
    
    For each data point in x, we will have a kernel value for each bin (i.e., a vector
    of K scalars)
    
    Parameters
    ----------
    x: Tensor
        Input to bin with shape (B, [C], N) (e.g., a flattened image).
    centers: Tensor
        The centers of the histogram bins with shape (B, K).
    sigma: float
        The standard deviation of the Gaussian kernel.
    
    Parameters
    ----------
    Tensor
        The kernel values of the input tensor with shape (B, [C], N, K).
    """
    assert x.dim() in (2, 3), (
        "Input tensors must have 2 dimensions (B, N), or 3 (B, C, N)."
    )
    if x.dim() == 2: # add channel dimension
        x = x.unsqueeze(1) # shape: (B, C, N) 
    
    # add dimensions to x and centers to allow broadcasting
    x = x.unsqueeze(-1) # shape: (B, C, N, 1)
    centers = centers.unsqueeze(1).unsqueeze(1) # shape: (B, 1, 1, K)
    
    # compute kernel values
    residuals = x - centers # shape: (B, C, N, K)
    return torch.exp(-0.5 * (residuals / sigma).pow(2))


def _sigmoid_kernel_binning(x: Tensor, centers: Tensor, scale: float) -> Tensor:
    """Bin the input tensor using sigmoid kernels centered on the bin centers.
    
    For each data point in x, we will have a kernel value for each bin (i.e., a vector
    of K scalars).
    
    Parameters
    ----------
    x: Tensor
        Input to bin with shape (B, [C], N) (e.g., a flattened image).
    centers: Tensor
        The centers of the histogram bins with shape (B, K).
    scale: float
        The scaling factor of the sigmoid kernel.
    
    Parameters
    ----------
    Tensor
        The kernel values of the input tensor with shape (B, C, N, K).
    """
    assert x.dim() in (2, 3), (
        "Input tensors must have 2 dimensions (B, N), or 3 (B, C, N)."
    )
    if x.dim() == 2: # add channel dimension
        x = x.unsqueeze(1) # shape: (B, C, N) 
    
    # add dimensions to x and centers to allow broadcasting
    x = x.unsqueeze(-1) # shape: (B, C, N, 1)
    centers = centers.unsqueeze(1).unsqueeze(1) # shape: (B, 1, 1, K)

    # compute kernel values
    delta = (centers[..., 1] - centers[..., 0])[..., None] # bin width (scalar)
    arg1 = scale * (x - (centers - delta / 2))
    arg2 = scale * (x - (centers + delta / 2))
    return torch.sigmoid(arg1) - torch.sigmoid(arg2)


def _kernel_binning(
    x: Tensor,
    centers: Tensor,
    method: Literal["gaussian", "sigmoid"],
    gaussian_sigma: Optional[float] = 0.5,
    sigmoid_scale: Optional[float] = 10.0
) -> Tensor:
    """Bin the input tensor using kernel functions centered on the bin centers.
    
    Parameters
    ----------
    x: Tensor
        Input to bin with shape (B, [C], N) (e.g., a flattened image).
    centers: Tensor
        The centers of the histogram bins with shape (B, K).
    method: Literal["gaussian", "sigmoid"]
        The method to compute the kernel binning.
    gaussian_sigma: float
        The standard deviation of the Gaussian kernel. Default is 0.5.
    sigmoid_scale: float
        The scaling factor of the sigmoid kernel. Default is 10.0.
        
    Returns
    -------
    Tensor
        The kernel values of the input tensor with shape (B, [C], N, K).
    """
    if method == "gaussian":
        kernel_values = _gaussian_kernel_binning(x, centers, gaussian_sigma)
    elif method == "sigmoid":
        kernel_values = _sigmoid_kernel_binning(x, centers, sigmoid_scale)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'gaussian' or 'sigmoid'.")
    
    return kernel_values


def _get_marginal_pdf(
    binned_values: Tensor, density: bool = True, epsilon: float = 1e-10
) -> Tensor:
    """Compute the marginal PDF from a variable's kernel binning values.
    
    Parameters
    ----------
    binned_values: Tensor
        The kernel binning values for the current batch with shape (B, [C], N, K).
    density: bool, optional
        Whether to normalize the histogram to get the PDF. Default is True.
    epsilon: float, optional
        A small value to avoid numerical instability. Default is 1e-10.
    
    Returns
    -------
    Tensor
        The marginal PDF of the variables in the current batch with shape (B, [C], K).
    """
    # average over samples to get the soft histogram
    hist = torch.mean(binned_values, dim=-2) # shape: (B, [C], K)
    
    # normalize to get the PDF (optional)
    normalization = 1
    if density:
        normalization = torch.sum(hist, dim=-1).unsqueeze(-1) + epsilon
    return hist / normalization


def _get_joint_pdf(
    binned_values1: Tensor,
    binned_values2: Tensor,
    density: bool = True,
    epsilon: float = 1e-10
) -> Tensor:
    """Compute the joint PDF from two variables kernel binning values.
    
    Parameters
    ----------
    binned_values1: Tensor
        The kernel binning values for the first variable and the current batch,
        with shape (B, [C], N, K).
    binned_values2: Tensor
        The kernel binning values for the second variable and the current batch,
        with shape (B, [C], N, K).
    density: bool, optional
        Whether to normalize the histogram to get the PDF. Default is True.
    epsilon: float, optional
        A small value to avoid numerical instability. Default is 1e-10.
    
    Returns
    -------
    Tensor
        The joint PDF of the variables in the current batch with shape (B, [C], K, K).
    """
    # calculate joint kernel values
    joint_kernel_values = torch.matmul(
        binned_values1.transpose(-2, -1), binned_values2
    ) # shape: (B, [C], K, K)

    # normalize to get the PDF (optional)
    normalization = 1
    if density:
        normalization = torch.sum(
            joint_kernel_values, dim=(-2, -1), keepdim=True
        ) + epsilon
    return joint_kernel_values / normalization


def _compute_mutual_info(
    marginal_1: Tensor, marginal_2: Tensor, joint: Tensor, epsilon: float = 1e-10
) -> float:
    """Compute the mutual information from marginal and joint distributions.
    
    This implementation uses the KL divergence definition of mutual information.
    
    Parameters
    ----------
    marginal_1: Tensor
        The marginal PDF of the first variable with shape (B, [C], K).
    marginal_2: Tensor
        The marginal PDF of the second variable with shape (B, [C], K).
    joint: Tensor
        The joint PDF of the two variables with shape (B, [C], K, K).
    epsilon: float, optional
        A small value to avoid log(0) and division by 0. Default is 1e-10.
    
    Returns
    -------
    float
        The mutual information between the two variables described by the PDFs.
    """
    marginal_prod = marginal_1.unsqueeze(-1) * marginal_2.unsqueeze(-2) # shape: (B, [C], K, K)
    joint[joint == 0.0] += epsilon # avoid log(0)
    marginal_prod[marginal_prod == 0.0] += epsilon # avoid log(0)
    return torch.sum(
        joint * (torch.log(joint) - torch.log(marginal_prod)), dim=(-2, -1)
    )


def soft_histogram(
    x: Tensor,
    centers: Tensor,
    method: Literal["gaussian", "sigmoid"],
    *,
    gaussian_sigma: Optional[float] = 0.5,
    sigmoid_scale: Optional[float] = 10.0,
    density: bool = True,
    epsilon: float = 1e-10
) -> Tensor:
    """Calculate the soft (differentiable) histogram (i.e., marginal PDF).
    
    The calculation uses either a Gaussian or a sigmoid kernel centered on the bin
    centers as an approximation of the indicator functions of the histogram bins.

    Parameters
    ----------
    x: Tensor
        Input to compute the histogram with shape (B, N) (e.g., a flattened image).
    centers: Tensor
        The centers of the histogram bins with shape (K).
    method: Literal["gaussian", "sigmoid"]
        The method to compute the soft histogram.
    gaussian_sigma: float, optional
        The standard deviation of the Gaussian kernel. A value in (0, 1] is
        recommended to get sharp binning functions. Default is 0.5.
    sigmoid_scale: float, optional
        The scaling factor of the sigmoid kernel. A value greater than 10 is
        recommended to get sharp binning functions. Default is 10.0.
    epsilon: float, optional
        A small value to avoid numerical instability. Default is 1e-10.

    Returns
    -------
    Tensor
        The soft histogram (marginal PDF) for the input tensor, shape is (B, K).
    """
    # bin input using kernel functions
    kernel_values = _kernel_binning(x, centers, method, gaussian_sigma, sigmoid_scale)

    # calculate histogram/PDF
    return _get_marginal_pdf(kernel_values, density, epsilon)


def soft_histogram2d(
    x1: Tensor,
    x2: Tensor,
    centers: Tensor,
    method: Literal["gaussian", "sigmoid"],
    *,
    gaussian_sigma: Optional[float] = 0.5,
    sigmoid_scale: Optional[float] = 10.0,
    density: bool = True,
    epsilon: float = 1e-10
) -> Tensor:
    """Calculate the 2d soft (differentiable) histogram (i.e., joint PDF).
    
    Parameters
    ----------
    x1: Tensor
        Input to compute the histogram with shape (B, N) (e.g., a flattened image).
    x2: Tensor
        Input to compute the histogram with shape (B, N) (e.g., a flattened image).
    centers: Tensor
        The centers of the histogram bins with shape (K).
    method: Literal["gaussian", "sigmoid"]
        The method to compute the soft histogram.
    gaussian_sigma: float, optional
        The standard deviation of the Gaussian kernel. A value in (0, 1] is
        recommended to get sharp binning functions. Default is 0.5.
    sigmoid_scale: float, optional
        The scaling factor of the sigmoid kernel. A value greater than 10 is
        recommended to get sharp binning functions. Default is 10.0.
    epsilon: float, optional
        A small value to avoid numerical instability. Default is 1e-10.
        
    Returns
    -------
    Tensor
        The 2d soft histogram (joint PDF) for the input tensors, shape is (B, K, K).
    """
    # bin input using kernel functions
    kernel_values1 = _kernel_binning(x1, centers, method, gaussian_sigma, sigmoid_scale)
    kernel_values2 = _kernel_binning(x2, centers, method, gaussian_sigma, sigmoid_scale)

    # calculate joint histogram/PDF
    return _get_joint_pdf(kernel_values1, kernel_values2, density, epsilon)


def mutual_information(
    x1: Tensor,
    x2: Tensor,
    num_bins: int,
    method: Literal["gaussian", "sigmoid"],
    gaussian_sigma: Optional[float] = 0.5,
    sigmoid_scale: Optional[float] = 10.0,
    epsilon: float = 1e-10
) -> float:
    """Calculate the (differentiable) mutual information between two input batches.

    This implementation uses the KL divergence definition of mutual information.

    Parameters
    ----------
    x1: Tensor
        Input tensor with shape (B, N) (e.g., a flattened image).
    x2: Tensor
        Input tensor with shape (B, N) (e.g., a flattened image).
    num_bins: int
        The number of bins to use for the histogram.
    method: Literal["gaussian", "sigmoid"]
        The method to compute the soft histogram.
    gaussian_sigma: float, optional
        The standard deviation of the Gaussian kernel. A value in (0, 1] is
        recommended to get sharp binning functions. Default is 0.5.
    sigmoid_scale: float, optional
        The scaling factor of the sigmoid kernel. A value greater than 10 is
        recommended to get sharp binning functions. Default is 10.0.
    epsilon: float, optional
        A small value to avoid numerical instability. Default is 1e-10.

    Returns
    -------
    float
        The mutual information between the two input batches.
    """
    # calculate the bin centers
    min_ = torch.min(torch.min(x1), torch.min(x2))
    max_ = torch.max(torch.max(x1), torch.max(x2))
    bins = torch.linspace(min_, max_, num_bins + 1, device=x1.device)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # calculate the joint PDF
    joint_pdf = soft_histogram2d(
        x1, x2, bin_centers, method,
        gaussian_sigma=gaussian_sigma,
        sigmoid_scale=sigmoid_scale,
        density=True,
        epsilon=epsilon
    )

    # calculate the marginal PDFs
    marginal_pdf1 = soft_histogram(
        x1, bin_centers, method, 
        gaussian_sigma=gaussian_sigma,
        sigmoid_scale=sigmoid_scale,
        density=True,
        epsilon=epsilon
    )
    marginal_pdf2 = soft_histogram(
        x2, bin_centers, method,
        gaussian_sigma=gaussian_sigma,
        sigmoid_scale=sigmoid_scale,
        density=True,
        epsilon=epsilon
    )

    # calculate the mutual information (using KL definition)
    return _compute_mutual_info(marginal_pdf1, marginal_pdf2, joint_pdf)
    #TODO: implement normalized mutual information -> in [0, 1], more interpretable...

# TODO: check if it works for a batch of elements
def pairwise_mutual_information(
    inputs: Tensor,
    num_bins: int,
    method: Literal["gaussian", "sigmoid"],
    gaussian_sigma: Optional[float] = 0.5,
    sigmoid_scale: Optional[float] = 10.0,
    epsilon: float = 1e-10   
) -> list[float]:
    """Calculate the (differentiable) pairwise mutual information between input
    channels.

    Parameters
    ----------
    inputs: Tensor
        Input tensor with shape (B, C, Z, Y, X).
    num_bins: int
        The number of bins to use for the histogram.
    method: Literal["gaussian", "sigmoid"]
        The method to compute the soft histogram.
    gaussian_sigma: float, optional
        The standard deviation of the Gaussian kernel. A value in (0, 1] is
        recommended to get sharp binning functions. Default is 0.5.
    sigmoid_scale: float, optional
        The scaling factor of the sigmoid kernel. A value greater than 10 is
        recommended to get sharp binning functions. Default is 10.0.
    epsilon: float, optional
        A small value to avoid numerical instability. Default is 1e-10.
        
    Returns
    -------
    torch.Tensor
        The pairwise mutual information between input channels over a batch.
        Hence, the output tensor has shape (B, C * (C - 1) / 2).
    """
    B, C, *spatial_dims = inputs.shape

    # calculate the bin centers (same for all channels, diff for each batch item)
    mins = torch.amin(inputs, dim=tuple(np.arange(1, len(spatial_dims) + 2)))
    maxs = torch.amax(inputs, dim=tuple(np.arange(1, len(spatial_dims) + 2)))
    bins = torch.stack([
        torch.linspace(min_, max_, num_bins + 1, device=inputs[:, 0].device)
        for min_, max_ in zip(mins, maxs)
    ], dim=0) # shape: (B, K + 1)
    bin_centers = (bins[:, :-1] + bins[:, 1:]) / 2 # shape: (B, K)
    
    # Create binnings for each channel
    binned_values = _kernel_binning(
        inputs.flatten(2),
        bin_centers,
        method,
        gaussian_sigma=gaussian_sigma,
        sigmoid_scale=sigmoid_scale
    ) # shape: (B, C, N, K), K=num_bins, N=[Z]*Y*X
    
    # Create marginal PDFs for each channel
    marginal_pdfs = _get_marginal_pdf(binned_values, epsilon) # shape: (B, C, K)
    
    # TODO: need to reshape binned_values with all pairwise combinations of channels
    # Get all pairwise joint pdfs
    joint_pdfs = {
        f"{i}_{j}" : _get_joint_pdf(binned_values[:, i], binned_values[:, j], epsilon)
        for i in range(C)
        for j in range(i + 1, C)
    } # shape: {C * C * (B, K, K)}
    
    # TODO: vectorize!
    # Calculate the pairwise mutual information
    return torch.stack([
        _compute_mutual_info(
            marginal_pdfs[:, i], marginal_pdfs[:, j], joint_pdfs[f"{i}_{j}"]
        )
        for i in range(C)
        for j in range(i + 1, C)
    ]).transpose(1, 0) # shape: (B, C * (C - 1) / 2)