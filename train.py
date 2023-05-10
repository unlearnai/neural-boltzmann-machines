import argparse
import math
from pathlib import Path

import torch
from lightning import Callback, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import CSVLogger

from datasets import DataModule
from neural_boltzmann_machine import NBM
from utils import PlotSamples, accelerator


class TrainableNBM(LightningModule):
    """
    Converts underlying Neural Boltzmann Machine class to a Lightning Module
    """

    def __init__(
        self,
        image_shape,
        num_classes,
        latent_size,
        visible_unit_type,
        mc_steps,
        bias_net_weight_decay,
        precision_net_weight_decay,
        weights_net_weight_decay,
        learning_rate,
        logger_dir,
    ):
        """
        Args:
            image_shape: Shape of images (tuple of int x int)
            num_classes: Number of classes in dataset (int)
            latent_size: Size of the latent space (int)
            visible_unit_type: type of visible units of (str)
            mc_steps: Number of MCMC sampling steps (int)
            bias_net_weight_decay: L2 penalty to apply to bias network (float)
            precision_net_weight_decay: L2 penalty to apply to precision network (float)
            weights_net_weight_decay: L2 penalty to apply to weights network (float)
            learning_rate: learning rate (float)
            logger_dir: directory where outputs should get logged (str)
        """
        super().__init__()

        self.image_shape = image_shape
        self.num_classes = num_classes
        self.latent_size = latent_size
        self.mc_steps = mc_steps
        self.bias_net_weight_decay = bias_net_weight_decay
        self.precision_net_weight_decay = precision_net_weight_decay
        self.weights_net_weight_decay = weights_net_weight_decay
        self.learning_rate = learning_rate
        self.logger_dir = logger_dir

        self.model = NBM(
            nx=num_classes,
            ny=math.prod(image_shape),
            nh=latent_size,
            visible_unit_type=visible_unit_type
        )
        self.save_hyperparameters()

    def training_step(self, batch, batch_index):
        """Train on CD loss."""
        y, x = batch
        losses = self.model.compute_loss(y, x, mc_steps=self.mc_steps)
        self.log("CD_Loss", losses["CD_loss"], prog_bar=True)
        self.log("MSE_Bias_Loss", losses["MSE_bias_loss"], prog_bar=True)
        self.log("MSE_Var_Loss", losses["MSE_variance_loss"], prog_bar=True)
        return losses["CD_loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.model.bias_net.parameters(),
                    "weight_decay": self.bias_net_weight_decay,
                },
                {
                    "params": self.model.precision_net.parameters(),
                    "weight_decay": self.precision_net_weight_decay,
                },
                {
                    "params": self.model.weights_net.parameters(),
                    "weight_decay": self.weights_net_weight_decay,
                },
            ],
            lr=self.learning_rate,
        )
        return [optimizer]

    def configure_callbacks(self):
        return PlotSamples(
            self.logger_dir, self.image_shape, self.num_classes, denoise=True
        )


def run(
    results_dir=".experiment_results",
    dataset_name="MNIST",
    visible_unit_type="gaussian",
    latent_size=64,
    max_epochs=50,
    learning_rate=5e-4,
    batch_size=128,
    num_workers=5,
    mc_steps=32,
    bias_net_weight_decay=0.5,
    precision_net_weight_decay=0.5,
    weights_net_weight_decay=1.0,
    log_interval=5,
    seed=1,
):
    """

    Args:
        results_dir: root directory where all results should be stored (str)
        dataset_name: name of dataset (str)
        visible_unit_type: type of visible units of (str)
        latent_size: Size of the latent space (int)
        max_epochs: maximum number of training epochs (int)
        learning_rate: learning rate (float)
        batch_size: size of each batch during training (int)
        num_workers: number of workers to use for dataloader (int)
        mc_steps: Number of MCMC sampling steps (int)
        bias_net_weight_decay: L2 penalty to apply to bias network (float)
        precision_net_weight_decay: L2 penalty to apply to precision network (float)
        weights_net_weight_decay: L2 penalty to apply to weights network (float)
        log_interval: what interval of iterations to log during each epoch (int)
        seed: what random seed to use (int)

    Returns:

    """

    # Prepare paths for data and results
    results_dir = Path(results_dir)
    datasets_dir = results_dir / "datasets"
    logger_dir = results_dir / "logs"

    # Setup logger and callbacks
    model_summary = ModelSummary(max_depth=1000)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = CSVLogger(logger_dir)
    # This is where this specific experiment run will be stored
    exp_version_dir = Path(logger.log_dir)
    checkpoint_dir = exp_version_dir / "checkpoints"
    model_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir, every_n_epochs=1, save_top_k=-1
    )

    # Setup the trainer
    seed_everything(seed)
    trainer = Trainer(
        accelerator=accelerator(),
        max_epochs=max_epochs,
        log_every_n_steps=log_interval,
        default_root_dir=logger_dir,
        logger=logger,
        callbacks=[lr_monitor, model_checkpoint, model_summary],
    )

    # Construct our dataset
    datamodule = DataModule(
        data_dir=str(datasets_dir),
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_name=dataset_name,
        output_type=visible_unit_type
    )

    # Construct our model
    model = TrainableNBM(
        image_shape=datamodule.dims,
        num_classes=datamodule.num_classes,
        latent_size=latent_size,
        visible_unit_type=datamodule.output_type,
        mc_steps=mc_steps,
        bias_net_weight_decay=bias_net_weight_decay,
        precision_net_weight_decay=precision_net_weight_decay,
        weights_net_weight_decay=weights_net_weight_decay,
        learning_rate=learning_rate,
        logger_dir=exp_version_dir,
    )

    # Train it!
    trainer.fit(model, datamodule)

    return trainer, model, datamodule, model.model


if __name__ == "__main__":
    """
    We have a simple command line interface to reproduce core results.

    We just expose two options for the the Dataset name and visible unit type.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        help="Should it be MNIST or FashionMNIST?",
        nargs="?",
        type=str,
        const=1,
        default="MNIST",
    )
    parser.add_argument(
        "--visible_unit_type",
        help="Should visible units be Ising or Gaussian?",
        nargs="?",
        type=str,
        const=1,
        default="gaussian",
    )

    args = vars(parser.parse_args())
    vis_unit = args["visible_unit_type"]
    if vis_unit == "gaussian":
        args["learning_rate"] = 5e-4
    elif vis_unit == "ising":
        args["learning_rate"] = 1e-2
    else:
        raise ValueError(f"Unrecognized visible unit type: {vis_unit}")

    lightning_trainer, lightning_model, lightning_datamodule, model = run(**args)
