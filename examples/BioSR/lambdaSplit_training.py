import argparse
import os
import glob
import json
from pathlib import Path
import socket
from typing import Literal, Optional

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
from careamics.utils.io_utils import get_git_status, get_workdir


# Model Parameters
loss_type: Optional[Literal["musplit", "denoisplit", "denoisplit_musplit", "lambdasplit"]] = "lambdasplit"
"""The type of reconstruction loss (i.e., likelihood) to use."""

# Data Parameters
batch_size: int = 32
"""The batch size for training."""
patch_size: list[int] = [64, 64]
"""Spatial size of the input patches."""
norm_strategy: Literal["channel-wise", "global"] = "channel-wise"
"""Normalization strategy for the input data."""

# Training Parameters
lr: float = 1e-3
"""The learning rate for training."""
lr_scheduler_patience: int = 30
"""The patience for the learning rate scheduler."""
earlystop_patience: int = 200
"""The patience for the learning rate scheduler."""
max_epochs: int = 400
"""The maximum number of epochs to train for."""
num_workers: int = 4
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
    img_size: int,
    target_ch: int,
    fluorophores: list[str],
    num_bins: int,
    wavelength_range: tuple[int, int],
    NM_paths: Optional[list[Path]] = None,
    training_config: TrainingConfig = TrainingConfig(),
    data_mean: Optional[torch.Tensor] = None,
    data_std: Optional[torch.Tensor] = None, 
) -> VAEModule:
    """Instantiate the lambdaSplit lightining model."""
    # Model config
    lvae_config = LVAEModel(
        architecture="LVAE",
        algorithm_type="unsupervised",
        input_shape=img_size,
        multiscale_count=1,
        z_dims=[128, 128, 128, 128],
        output_channels=target_ch,
        predict_logvar=None,
        analytical_kl=False,
        fluorophores=fluorophores,
        wv_range=wavelength_range,
        num_bins=num_bins,
        ref_learnable=False,
    )
    
    # Loss config
    kl_loss_config = KLLossConfig(
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
) -> None:
    """Train the lambdaSplit model.
    
    Parameters
    ----------
    root_dir : str
        The root directory where the training results will be saved.
    data_dir : str
        The directory where the data is stored.
    """
    # Load metadata
    with open(os.path.join(data_dir, "sim_metadata.json"), "r") as f:
        metadata = json.load(f)
    F = len(metadata["fluorophores"])
    N = metadata["num_bins"]
    
    # Set working directory
    algo = "lambdasplit"
    workdir, exp_tag = get_workdir(root_dir, f"{algo}_BioSR_F{F}_N{N}")
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
        axes="CYX",
        patch_size=patch_size,
        batch_size=batch_size,
        transforms=[],
        norm_strategy=norm_strategy,
    )
    
    # Load data
    path_to_imgs = os.path.join(data_dir, "imgs/digital")
    fnames = [
        Path(path_to_imgs) / fname 
        for fname in glob.glob(os.path.join(path_to_imgs, "*.tif"))
    ]
    train_dset = InMemoryDataset(
        data_config=data_config,
        inputs=fnames,
    )
    val_dset = train_dset.split_dataset(percentage=0.15)

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
        img_size=patch_size[0],
        target_ch=len(metadata["fluorophores"]),
        fluorophores=metadata["fluorophores"],
        training_config=training_config,
        num_bins=metadata["num_bins"],
        wavelength_range=metadata["wavelength_range"],
    )
    
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
    with open(os.path.join(workdir, "sim_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Save Configs in WanDB
    custom_logger.experiment.config.update({"algorithm": algo_config.model_dump()})
    custom_logger.experiment.config.update({"training": training_config.model_dump()})
    custom_logger.experiment.config.update({"data": data_config.model_dump()})
    custom_logger.experiment.config.update({"sim_metadata": metadata})
    
    # Define callbacks (e.g., ModelCheckpoint, EarlyStopping, etc.)
    custom_callbacks = [
        EarlyStopping(
            monitor="val_loss",
            min_delta=1e-6,
            patience=training_config.earlystop_patience,
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=workdir,
            filename="best-{epoch}",
            monitor="val_loss",
            save_top_k=1,
            save_last=True,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str)
    parser.add_argument("--data-dir", type=str)
    args = parser.parse_args()
    train(root_dir=args.root_dir, data_dir=args.data_dir)