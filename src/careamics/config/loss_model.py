"""Configuration classes for LVAE losses."""

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, model_validator


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
    current_epoch: int = 0
    """Current epoch in the training loop."""


class LVAELossConfig(BaseModel):
    """LVAE loss configuration."""

    model_config = ConfigDict(
        validate_assignment=True, validate_default=True, arbitrary_types_allowed=True
    )

    loss_type: Literal["musplit", "denoisplit", "denoisplit_musplit"]
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
    kl_params: Union[KLLossConfig, dict[str, KLLossConfig]]
    """KL loss configuration."""

    # TODO: remove?
    non_stochastic: bool = False
    """Whether to sample latents and compute KL."""
    
    @model_validator(mode="after")
    def validate_kl_params(self) -> None:
        """Validate the KL parameters."""
        if self.loss_type == "musplit":
            assert isinstance(self.kl_params, KLLossConfig)
        elif self.loss_type == "denoisplit":
            assert isinstance(self.kl_params, KLLossConfig)
        elif self.loss_type == "denoisplit_musplit":
            assert isinstance(self.kl_params, dict), (
                "With 'denoisplit_musplit' loss, kl_params must be a dictionary",
                "with keys 'denoisplit' and 'musplit' and corresponding KLLossConfig's",
                "as values."
            )
            assert set(self.kl_params.keys()) == set(["denoisplit", "musplit"])
            assert isinstance(self.kl_params["denoisplit"], KLLossConfig)
            assert isinstance(self.kl_params["musplit"], KLLossConfig)
            
        else:
            raise ValueError(f"Unknown loss type {self.loss_type}.")
        
        return self
