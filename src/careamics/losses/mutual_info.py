"""Implementation of differentiable histogram functions for the computation of mutual information."""

from typing import Literal, Optional

import torch
from torch import Tensor


def _gaussian_kernel_binning(
    x: Tensor,
    centers: Tensor,
    sigma: float    
) -> Tensor:
    """Bin the input tensor using Gaussian kernels centered on the bin centers.
    
    For each data point in x, we will have a kernel value for each bin (i.e., a vector
    of K scalars)
    
    Parameters
    ----------
    x: Tensor
        Input to bin with shape (B, N) (e.g., a flattened image).
    centers: Tensor
        The centers of the histogram bins with shape (K).
    sigma: float
        The standard deviation of the Gaussian kernel.
    
    Parameters
    ----------
    Tensor
        The kernel values of the input tensor with shape (B, N, K).
    """
    # add dimensions to x and centers to allow broadcasting
    x = x.unsqueeze(2)
    centers = centers.unsqueeze(0).unsqueeze(0)
    
    # compute kernel values
    residuals = x - centers
    return torch.exp(-0.5 * (residuals / sigma).pow(2))


def _sigmoid_kernel_binning(
    x: Tensor,
    centers: Tensor,
    scale: float
) -> Tensor:
    """Bin the input tensor using sigmoid kernels centered on the bin centers.
    
    For each data point in x, we will have a kernel value for each bin (i.e., a vector
    of K scalars).
    
    Parameters
    ----------
    x: Tensor
        Input to bin with shape (B, N) (e.g., a flattened image).
    centers: Tensor
        The centers of the histogram bins with shape (K).
    scale: float
        The scaling factor of the sigmoid kernel.
    
    Parameters
    ----------
    Tensor
        The kernel values of the input tensor with shape (B, N, K).
    """
    # add dimensions to x and centers to allow broadcasting
    x = x.unsqueeze(2)
    centers = centers.unsqueeze(0).unsqueeze(0)
    
    # compute kernel values
    delta = centers[..., 1] - centers[..., 0] # bin width
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
        Input to bin with shape (B, N) (e.g., a flattened image).
    centers: Tensor
        The centers of the histogram bins with shape (K).
    method: Literal["gaussian", "sigmoid"]
        The method to compute the kernel binning.
    gaussian_sigma: float
        The standard deviation of the Gaussian kernel. Default is 0.5.
    sigmoid_scale: float
        The scaling factor of the sigmoid kernel. Default is 10.0.
    """
    if method == "gaussian":
        kernel_values = _gaussian_kernel_binning(x, centers, gaussian_sigma)
    elif method == "sigmoid":
        kernel_values = _sigmoid_kernel_binning(x, centers, sigmoid_scale)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'gaussian' or 'sigmoid'.")
    
    return kernel_values


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

    # average over samples to get the soft histogram
    hist = torch.mean(kernel_values, dim=1)
    
    # normalize to get the PDF (optional)
    if density:
        normalization = torch.sum(hist, dim=1).unsqueeze(1) + epsilon
        return hist / normalization

    return hist


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

    # calculate joint PDF
    joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
    
    # normalize to get the PDF (optional)
    if density:
        normalization = torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1) + epsilon
        return joint_kernel_values / normalization

    return joint_kernel_values


def mutual_information(
    x1: Tensor,
    x2: Tensor,
    num_bins: int,
    method: Literal["gaussian", "sigmoid"],
    gaussian_sigma: Optional[float] = 0.5,
    sigmoid_scale: Optional[float] = 10.0,
    epsilon: float = 1e-10
) -> Tensor:
    """Calculate the (differentiable) mutual information between two input tensors.
    
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
    Tensor
        The mutual information between the two input tensors, shape is (B).
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
    return torch.sum(
        joint_pdf * (
            torch.log(joint_pdf + epsilon) - 
            torch.log(marginal_pdf1 + epsilon) - 
            torch.log(marginal_pdf2 + epsilon)
        ), 
        dim=(1, 2)
    )