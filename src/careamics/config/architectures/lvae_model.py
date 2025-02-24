"""LVAE Pydantic model."""

from typing import Literal, Optional

from pydantic import ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from .architecture_model import ArchitectureModel


class LVAEModel(ArchitectureModel):
    """LVAE model."""

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    architecture: Literal["LVAE"]
    
    training_mode: Literal["unsupervised", "supervised"]
    """Training mode for the LVAE model."""
    input_shape: list[int] = Field(default=(64, 64), validate_default=True)
    """Shape of the input patch (C, Z, Y, X) or (C, Y, X) if the data is 2D."""
    encoder_conv_strides: list = Field(default=[2, 2], validate_default=True)
    # TODO make this per hierarchy step ?
    decoder_conv_strides: list = Field(default=[2, 2], validate_default=True)
    """Dimensions (2D or 3D) of the convolutional layers."""
    multiscale_count: int = Field(default=1)
    # TODO there should be a check for multiscale_count in dataset !!

    # 1 - off, len(z_dims) + 1 # TODO Consider starting from 0
    z_dims: list = Field(default=[128, 128, 128, 128])
    output_channels: int = Field(default=1, ge=1)
    encoder_n_filters: int = Field(default=64, ge=8, le=1024)
    decoder_n_filters: int = Field(default=64, ge=8, le=1024)
    encoder_dropout: float = Field(default=0.1, ge=0.0, le=0.9)
    decoder_dropout: float = Field(default=0.1, ge=0.0, le=0.9)
    nonlinearity: Literal[
        "None", "Sigmoid", "Softmax", "Tanh", "ReLU", "LeakyReLU", "ELU"
    ] = Field(
        default="ELU",
    )
    predict_logvar: Literal[None, "pixelwise"] = None

    analytical_kl: bool = Field(default=False)
    
    # λSplit parameters
    fluorophores: list[str]
    """A list of the fluorophore names in the image to unmix."""
    wv_range: tuple[int, int] = Field(default=(400, 700))
    """The wavelength range of the spectral image."""
    num_bins: int = 32
    """Number of bins for the spectral data."""
    clip_unmixed: bool = False
    """Whether to clip negative values in the unmixed spectra to 0."""
    ref_learnable: bool = False
    """Whether the reference spectra matrix is learnable."""
    mixer_num_frozen_epochs: int = 0
    """Number of epochs before starting to learn the spectra reference matrix."""
    add_background: Optional[Literal["random", "constant", "from_image"]] = None
    """Whether and how to add a background spectrum to the reference matrix."""
    bg_learnable: bool = False
    """Whether the background spectrum is learnable."""
    bg_kwargs: Optional[dict] = None
    """Additional keyword arguments for the background spectrum."""
    
    @model_validator(mode="after")
    def validate_conv_strides(self: Self) -> Self:
        """
        Validate the convolutional strides.

        Returns
        -------
        list
            Validated strides.

        Raises
        ------
        ValueError
            If the number of strides is not 2.
        """
        if len(self.encoder_conv_strides) < 2 or len(self.encoder_conv_strides) > 3:
            raise ValueError(
                f"Strides must be 2 or 3 (got {len(self.encoder_conv_strides)})."
            )

        if len(self.decoder_conv_strides) < 2 or len(self.decoder_conv_strides) > 3:
            raise ValueError(
                f"Strides must be 2 or 3 (got {len(self.decoder_conv_strides)})."
            )

        # adding 1 to encoder strides for the number of input channels
        if len(self.input_shape) != len(self.encoder_conv_strides):
            raise ValueError(
                f"Input dimensions must be equal to the number of encoder conv strides"
                f" (got {len(self.input_shape)} and {len(self.encoder_conv_strides)})."
            )

        if len(self.encoder_conv_strides) < len(self.decoder_conv_strides):
            raise ValueError(
                f"Decoder can't be 3D when encoder is 2D (got"
                f" {len(self.encoder_conv_strides)} and"
                f"{len(self.decoder_conv_strides)})."
            )

        if any(s < 1 for s in self.encoder_conv_strides) or any(
            s < 1 for s in self.decoder_conv_strides
        ):
            raise ValueError(
                f"All strides must be greater or equal to 1"
                f"(got {self.encoder_conv_strides} and {self.decoder_conv_strides})."
            )
        # TODO: validate max stride size ?
        return self

    @field_validator("input_shape")
    @classmethod
    def validate_input_shape(cls, input_shape: list) -> list:
        """
        Validate the input shape.

        Parameters
        ----------
        input_shape : list
            Shape of the input patch.

        Returns
        -------
        list
            Validated input shape.

        Raises
        ------
        ValueError
            If the number of dimensions is not 3 or 4.
        """
        if len(input_shape) < 2 or len(input_shape) > 3:
            raise ValueError(
                f"Number of input dimensions must be 2 for 2D data 3 for 3D"
                f"(got {len(input_shape)})."
            )

        if any(s < 1 for s in input_shape):
            raise ValueError(
                f"Input shape must be greater than 1 in all dimensions"
                f"(got {input_shape})."
            )
        return input_shape

    @field_validator("encoder_n_filters")
    @classmethod
    def validate_encoder_even(cls, encoder_n_filters: int) -> int:
        """
        Validate that num_channels_init is even.

        Parameters
        ----------
        encoder_n_filters : int
            Number of channels.

        Returns
        -------
        int
            Validated number of channels.

        Raises
        ------
        ValueError
            If the number of channels is odd.
        """
        # if odd
        if encoder_n_filters % 2 != 0:
            raise ValueError(
                f"Number of channels for the bottom layer must be even"
                f" (got {encoder_n_filters})."
            )

        return encoder_n_filters

    @field_validator("decoder_n_filters")
    @classmethod
    def validate_decoder_even(cls, decoder_n_filters: int) -> int:
        """
        Validate that num_channels_init is even.

        Parameters
        ----------
        decoder_n_filters : int
            Number of channels.

        Returns
        -------
        int
            Validated number of channels.

        Raises
        ------
        ValueError
            If the number of channels is odd.
        """
        # if odd
        if decoder_n_filters % 2 != 0:
            raise ValueError(
                f"Number of channels for the bottom layer must be even"
                f" (got {decoder_n_filters})."
            )

        return decoder_n_filters

    @field_validator("z_dims")
    def validate_z_dims(cls, z_dims: tuple) -> tuple:
        """
        Validate the z_dims.

        Parameters
        ----------
        z_dims : tuple
            Tuple of z dimensions.

        Returns
        -------
        tuple
            Validated z dimensions.

        Raises
        ------
        ValueError
            If the number of z dimensions is not 4.
        """
        if len(z_dims) < 2:
            raise ValueError(
                f"Number of z dimensions must be at least 2 (got {len(z_dims)})."
            )

        return z_dims

    @model_validator(mode="after")
    def validate_multiscale_count(self: Self) -> Self:
        """
        Validate the multiscale count.

        Returns
        -------
        Self
            The validated model.
        """
        if self.multiscale_count < 1 or self.multiscale_count > len(self.z_dims) + 1:
            raise ValueError(
                f"Multiscale count must be 1 for LC off or less or equal to the number"
                f" of Z dims + 1 (got {self.multiscale_count} and {len(self.z_dims)})."
            )
        return self
    
    @model_validator(mode="after")
    def _validate_output_channels(self: Self) -> Self:
        """
        Validate the output channels.

        Returns
        -------
        Self
            The validated model.
        """
        if self.training_mode == "unsupervised":
            if self.fluorophores:
                if self.add_background:
                    if self.output_channels != len(self.fluorophores) + 1:
                        raise ValueError(
                            f"Output channels must be equal to the number of "
                            "fluorophores plus one for the background (got "
                            f"{self.output_channels} and {len(self.fluorophores)})."
                        )
                else:
                    if self.output_channels != len(self.fluorophores):
                        raise ValueError(
                            f"Output channels must be equal to the number of "
                            f"fluorophores (got {self.output_channels} and "
                            f"{len(self.fluorophores)})."
                        )
        return self
    
    @model_validator(mode="after")
    def _validate_background_args(self: Self) -> Self:
        """
        Validate the background arguments.

        Returns
        -------
        Self
            The validated model.
        """
        if self.add_background == "from_image":
            if self.bg_kwargs is None:
                raise ValueError(
                    "Background kwargs must be provided if background spectrum needs"
                    "to be extracted from the image."
                )
            else:
                if "image" not in self.bg_kwargs:
                    raise ValueError(
                        "`image` must be provided in the background kwargs."
                    )
                if "coords" not in self.bg_kwargs:
                    raise ValueError(
                        "`coords` must be provided in the background kwargs."
                    )
        else:
            if self.bg_kwargs is None:
                self.bg_kwargs = {}
        return self