"""Configuration classes for LVAE losses."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class KLLossConfig(BaseModel):
    """KL loss configuration."""

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    loss_type: Literal["kl", "kl_restricted"] = "kl"
    """Type of KL divergence used as KL loss."""
    rescaling: Literal["latent_dim", "image_dim"] = "latent_dim"
    """Rescaling of the KL loss."""
    aggregation: Literal["sum", "mean"] = "mean"
    """Aggregation of the KL loss across different layers."""
    free_bits_coeff: float = 0.0
    """Free bits coefficient for the KL loss."""
    annealing: bool = False
    """Whether to apply KL loss annealing."""
    start: int = -1
    """Epoch at which KL loss annealing starts."""
    annealtime: int = 10
    """Number of epochs for which KL loss annealing is applied."""
    current_epoch: int = 0 # TODO: done by lightning, remove (?)
    """Current epoch in the training loop."""
    

class MutualInfoLossConfig(BaseModel):
    """Mutual information loss configuration."""

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    loss_type: Literal["hist", "MINE"] = "hist"
    """Type of mutual information implementation, either using histograms to estimate
    joint and marginal distributions of input data or using MINE algorithm."""
    num_bins: int = 20
    """Number of bins for the histogram approximating input distribution. A good rule
    of thumb is to set the number of bins approximately equal to 1/3 of the square root
    of the number of samples (e.g., for an image 100x100 a sensible values is ~30)."""
    binning_method: Literal["gaussian", "sigmoid"] = "sigmoid"
    """Methods for differentiable binning in the histogram approach."""
    gaussian_sigma: float = 0.1
    """The standard deviation of the Gaussian kernel. A value in (0, 1] is
    recommended to get sharper binning functions and, hence, a better estimate of the
    mutual information."""
    sigmoid_scale: float = 500.0
    """The scaling factor of the sigmoid kernel. A value greater than 10 is
    recommended to get sharper binning functions and, hence, a better estimate of the
    mutual information."""
    epsilon: float = 1e-10  
    """Small value to ensure numerical stability."""
    # TODO: add burn-in period before starting to compute mutual information
    
    @field_validator("loss_type")
    def validate_loss_type(cls, v):
        if v == "MINE":
            raise NotImplementedError("MINE mutual info loss not yet implemented")
        return v


class LVAELossConfig(BaseModel):
    """LVAE loss configuration."""

    model_config = ConfigDict(
        validate_assignment=True, validate_default=True, arbitrary_types_allowed=True
    )

    loss_type: Literal["musplit", "denoisplit", "denoisplit_musplit", "lambdasplit"]
    """Type of loss to use for LVAE."""

    reconstruction_weight: float = 1.0
    """Weight for the reconstruction loss in the total net loss
    (i.e., `net_loss = reconstruction_weight * rec_loss + kl_weight * kl_loss`)."""
    kl_weight: float = 1.0
    """Weight for the KL loss in the total net loss.
    (i.e., `net_loss = reconstruction_weight * rec_loss + kl_weight * kl_loss`)."""
    musplit_weight: float = 0.1
    """Weight for the muSplit loss (used in the muSplit-denoiSplit loss)."""
    denoisplit_weight: float = 0.9
    """Weight for the denoiSplit loss (used in the muSplit-deonoiSplit loss)."""
    kl_params: KLLossConfig = KLLossConfig()
    """KL loss configuration."""
    mutual_info_weight: float = 1.0
    """Weight for the mutual information loss."""
    mutual_info_params: MutualInfoLossConfig = MutualInfoLossConfig()
    """Mutual information loss configuration."""

    # TODO: remove?
    non_stochastic: bool = False
    """Whether to sample latents and compute KL."""