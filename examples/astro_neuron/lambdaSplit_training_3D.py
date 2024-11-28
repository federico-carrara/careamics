import argparse
import os
import json
from pathlib import Path
import socket
from typing import Any, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
import torch
from torch.utils.data import DataLoader
import wandb

from careamics.config import DataConfig, TrainingConfig
from careamics.config import VAEAlgorithmConfig
from careamics.config.architectures import LVAEModel
from careamics.config.likelihood_model import (
    GaussianLikelihoodConfig,
    NMLikelihoodConfig,
)
from careamics.config.loss_model import LVAELossConfig, KLLossConfig
from careamics.config.nm_model import GaussianMixtureNMConfig, MultiChannelNMConfig
from careamics.config.optimizer_models import LrSchedulerModel, OptimizerModel
from careamics.lightning import VAEModule
from careamics.dataset import InMemoryDataset
from careamics.dataset.dataset_utils.readers.astro_neurons import (
    get_fnames, get_train_test_fnames, get_max_z_size, load_3D_img
)
from careamics.utils.io_utils import get_git_status, get_workdir


# pydantic model for additional λSplit parameters
# TODO: move this to the experiment repo when it will be there
class ExtraLambdaParameters(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True, validate_default=True, extra="allow"
    )
    dset_type: Literal["astrocytes", "neurons"]
    """The type of dataset to use."""
    img_type : Literal["raw", "unmixed"]
    """The type of image to load, i.e., either raw multispectral or unmixed stacks."""
    groups : Sequence[Literal["control", "arsenite", "tharps"]]
    """The groups of samples to load."""
    dim : Literal["2D", "3D"] = "2D"
    """The dimensionality of the images to load."""


# General Parameters
loss_type: Optional[Literal["musplit", "denoisplit", "denoisplit_musplit", "lambdasplit"]] = "lambdasplit"
"""The type of reconstruction loss (i.e., likelihood) to use."""
batch_size: int = 32
"""The batch size for training."""
patch_size: list[int] = [8, 64, 64]
"""Spatial size of the input patches."""
norm_strategy: Literal["channel-wise", "global"] = "channel-wise"
"""Normalization strategy for the input data."""

# λSplit Parameters
lambda_params = ExtraLambdaParameters(
    dset_type="astrocytes",
    img_type="raw",
    groups=["control"],
    dim="3D",
)

# Training Parameters
lr: float = 1e-3
"""The learning rate for training."""
lr_scheduler_patience: int = 30
"""The patience for the learning rate scheduler."""
earlystop_patience: int = 200
"""The patience for the learning rate scheduler."""
max_epochs: int = 400
"""The maximum number of epochs to train for."""
num_workers: int = 3
"""The number of workers to use for data loading."""

# Additional not to touch parameters
multiscale_count: int = 1
"""The number of LC inputs plus one (the actual input)."""
predict_logvar: Optional[Literal["pixelwise"]] = None
"""Whether to compute also the log-variance as LVAE output."""
nm_paths: Optional[tuple[str]] = [
    "/group/jug/ashesh/training_pre_eccv/noise_model/2402/221/GMMNoiseModel_ER-GT_all.mrc__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz",
    "/group/jug/ashesh/training_pre_eccv/noise_model/2402/225/GMMNoiseModel_Microtubules-GT_all.mrc__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz",
]
"""The paths to the pre-trained noise models for the different channels."""


def unsupervised_collate_fn(batch: list[torch.Tensor, None]) -> torch.Tensor:
    inputs = [item[0] for item in batch]
    inputs = torch.stack([torch.from_numpy(input_array) for input_array in inputs], dim=0)
    return inputs, None

def create_lambda_split_lightning_model(
    algorithm: str,
    loss_type: str,
    img_size: tuple[int, int],
    spectral_metadata: dict[str, Any],
    NM_paths: Optional[list[Path]] = None,
    training_config: TrainingConfig = TrainingConfig(),
    data_mean: Optional[torch.Tensor] = None,
    data_std: Optional[torch.Tensor] = None, 
) -> VAEModule:
    """Instantiate the lambdaSplit lightining model."""
    # Model config
    lvae_config = LVAEModel(
        architecture="LVAE",
        training_mode="unsupervised",
        input_shape=img_size,
        encoder_conv_strides=[1, 2, 2], # TODO: this chnages in 3D case
        decoder_conv_strides=[1, 2, 2], # TODO: this changes in 3D case
        multiscale_count=1,
        z_dims=[128, 128, 128, 128],
        output_channels=len(spectral_metadata["fluorophores"]),
        predict_logvar=None,
        analytical_kl=False,
        fluorophores=spectral_metadata["fluorophores"],
        wv_range=spectral_metadata["wavelength_range"],
        num_bins=spectral_metadata["num_bins"],
        ref_learnable=False,
    )
    
    # Loss config
    kl_loss_config = KLLossConfig(
        loss_type="kl",
        rescaling="latent_dim",
        aggregation="mean",
        free_bits_coeff=0.0,
    )
    loss_config = LVAELossConfig(
        loss_type=loss_type,
        kl_params=kl_loss_config.model_dump(), # TODO: why tf needs model dump?
    )

    # Likelihoods configs
    # gaussian likelihood
    if loss_type in ["musplit", "denoisplit_musplit", "lambdasplit"]:
        gaussian_lik_config = GaussianLikelihoodConfig(
            predict_logvar=predict_logvar,
            logvar_lowerbound=-5.0,  # TODO: find a better way to fix this
        )
    else:
        gaussian_lik_config = None
    # noise model likelihood
    if loss_type in ["denoisplit", "denoisplit_musplit"]:
        assert NM_paths is not None, "A path to a pre-trained noise model is required."
        gmm_list = []
        for NM_path in NM_paths:
            gmm_list.append(
                GaussianMixtureNMConfig(
                    model_type="GaussianMixtureNoiseModel",
                    path=NM_path,
                )
            )
        noise_model_config = MultiChannelNMConfig(noise_models=gmm_list)
        nm_lik_config = NMLikelihoodConfig(data_mean=data_mean, data_std=data_std)
    else:
        noise_model_config = None
        nm_lik_config = None

    # Other configs
    opt_config = OptimizerModel(
        name="Adamax",
        parameters={
            "lr": training_config.lr,
            "weight_decay": 0,
        },
    )
    lr_scheduler_config = LrSchedulerModel(
        name="ReduceLROnPlateau",
        parameters={
            "mode": "min",
            "factor": 0.5,
            "patience": training_config.lr_scheduler_patience,
            "verbose": True,
            "min_lr": 1e-12,
        },
    )
 
    # Group all configs & create model
    vae_config = VAEAlgorithmConfig(
        algorithm_type="vae",
        algorithm=algorithm,
        model=lvae_config,
        loss=loss_config.model_dump(), # TODO: why tf needs model dump?
        gaussian_likelihood=gaussian_lik_config,
        noise_model=noise_model_config,
        noise_model_likelihood=nm_lik_config,
        optimizer=opt_config,
        lr_scheduler=lr_scheduler_config,
    )
    return VAEModule(algorithm_config=vae_config)


def train(
    root_dir: str,
    data_dir: str,
    logging: bool = True,
) -> None:
    """Train the lambdaSplit model.
    
    Parameters
    ----------
    root_dir : str
        The root directory where the training results will be saved.
    data_dir : str
        The directory where the data is stored.
    logging : bool
        Whether to log the results and configs.
    """
    # Load metadata
    with open(os.path.join(data_dir, lambda_params.dset_type, "info/metadata.json")) as f:
        metadata = json.load(f)
    
    # Set working directory
    if logging:
        algo = "lambdasplit"
        workdir, exp_tag = get_workdir(root_dir, f"{algo}_{lambda_params.dset_type[:5]}")
        print(f"Current workdir: {workdir}")
    
    # Create configs
    training_config = TrainingConfig(
        num_epochs=max_epochs,
        precision="16-mixed",
        logger="wandb",
        gradient_clip_algorithm= "value",
        grad_clip_norm_value=0.5,
        lr=lr,
        lr_scheduler_patience=lr_scheduler_patience,
    )
    data_config = DataConfig(
        data_type="tiff",
        axes="CZYX", # TODO: this differs in 3D case
        patch_size=patch_size,
        batch_size=batch_size,
        transforms=[],
        norm_strategy=norm_strategy,
    )
    
    # Load data
    fnames = get_fnames(
        data_dir,
        dset_type=lambda_params.dset_type,
        img_type=lambda_params.img_type,
        groups=lambda_params.groups,
        dim=lambda_params.dim,
    )
    max_z = get_max_z_size(fnames)
    train_fnames, val_fnames = get_train_test_fnames(
        fnames, test_percent=0.1, deterministic=True
    )
    train_dset = InMemoryDataset(
        data_config=data_config,
        inputs=train_fnames,
        read_source_func=load_3D_img,
        read_source_kwargs={"max_z": max_z},
    )
    val_dset = InMemoryDataset(
        data_config=data_config,
        inputs=val_fnames,
        read_source_func=load_3D_img,
        read_source_kwargs={"max_z": max_z},
    )

    train_dloader = DataLoader(
        train_dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=unsupervised_collate_fn
    )
    val_dloader = DataLoader(
        val_dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers, 
        collate_fn=unsupervised_collate_fn
    )
    
    # Create model
    lightning_model = create_lambda_split_lightning_model(
        algorithm="lambdasplit",
        loss_type=loss_type,
        img_size=patch_size,
        spectral_metadata=metadata,
        training_config=training_config,
    )
    
    custom_logger = None
    if logging:
        # Define the logger
        custom_logger = WandbLogger(
            name=os.path.join(socket.gethostname(), exp_tag),
            save_dir=workdir,
            project="_".join(("careamics", algo)),
        )
        
        # Save configs and git status (for debugging)
        algo_config = lightning_model.algorithm_config
        with open(os.path.join(workdir, "git_config.json"), "w") as f:
            json.dump(get_git_status(), f, indent=4)
        with open(os.path.join(workdir, "algorithm_config.json"), "w") as f:
            f.write(algo_config.model_dump_json(indent=4))
        with open(os.path.join(workdir, "training_config.json"), "w") as f:
            f.write(training_config.model_dump_json(indent=4))
        with open(os.path.join(workdir, "data_config.json"), "w") as f:
            f.write(data_config.model_dump_json(indent=4))
        with open(os.path.join(workdir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        with open(os.path.join(workdir, "lambda_params.json"), "w") as f:
            f.write(lambda_params.model_dump_json(indent=4))
        
        # Save Configs in WanDB
        custom_logger.experiment.config.update({"algorithm": algo_config.model_dump()})
        custom_logger.experiment.config.update({"training": training_config.model_dump()})
        custom_logger.experiment.config.update({"data": data_config.model_dump()})
        custom_logger.experiment.config.update({"metadata": metadata})
        custom_logger.experiment.config.update({"lambda_params": lambda_params.model_dump()})
        
    # Define callbacks (e.g., ModelCheckpoint, EarlyStopping, etc.)
    custom_callbacks = [
        EarlyStopping(
            monitor="val_loss",
            min_delta=1e-6,
            patience=training_config.earlystop_patience,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if logging:
        custom_callbacks.append(
            ModelCheckpoint(
                dirpath=workdir,
                filename="best-{epoch}",
                monitor="val_loss",
                save_top_k=1,
                save_last=True,
                mode="min",
            ),
        )
    
    trainer = Trainer(
        max_epochs=training_config.num_epochs,
        accelerator="gpu",
        enable_progress_bar=True,
        logger=custom_logger,
        callbacks=custom_callbacks,
        precision=training_config.precision,
        gradient_clip_val=training_config.gradient_clip_val,  # only works with `accelerator="gpu"`
        gradient_clip_algorithm=training_config.gradient_clip_algorithm,
    )   
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_dloader,
        val_dataloaders=val_dloader,
    )
    wandb.finish()


if __name__ == "__main__":
    ROOT_DIR = "/group/jug/federico/lambdasplit_training/"
    DATA_DIR = "/group/jug/federico/data/neurons_and_astrocytes"
    parser = argparse.ArgumentParser(description="Train the lambdaSplit model.")
    parser.add_argument(
        "--no-log", "-n",
        action="store_true",
        help="Disable logging"
    )
    args = parser.parse_args()
    train(root_dir=ROOT_DIR, data_dir=DATA_DIR, logging=not args.no_log)