# Copyright Grantham Taylor.

import flytekit as fl

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar, LearningRateMonitor
from lightning.pytorch.loggers.wandb import WandbLogger

from windmark.core.architecture.encoders import SequenceModule
from windmark.core.constructs.managers import SystemManager
from windmark.core.constructs.general import Hyperparameters
from windmark.core.architecture.callbacks import ThawedFinetuning
from windmark.orchestration.environments import context


@context.lab
def finetune_sequence_encoder(
    lifestreams: fl.FlyteDirectory,
    checkpoint: fl.FlyteFile,
    params: Hyperparameters,
    manager: SystemManager,
    label: str,
) -> fl.FlyteFile:
    """
    Finetunes a pretrained sequence encoder model using the provided lifestreams data, checkpoint, hyperparameters, and system manager.

    Args:
        lifestreams (fl.FlyteDirectory): The directory containing the lifestreams data.
        checkpoint (fl.FlyteFile): The pretrained checkpoint file to load the initial model weights from.
        params (Hyperparameters): The hyperparameters for the finetuning process.
        manager (SystemManager): The system state manager.
        label (str): The name for the experiment.

    Returns:
        fl.FlyteFile: The file object representing the best model checkpoint after finetuning.
    """

    torch.set_float32_matmul_precision("medium")
    torch.multiprocessing.set_sharing_strategy("file_system")

    module = SequenceModule.load_from_checkpoint(
        checkpoint_path=str(checkpoint.path),
        datapath=str(lifestreams.path),
        params=params,
        manager=manager,
        mode="finetune",
    )

    checkpointer = ModelCheckpoint(
        dirpath=fl.current_context().working_directory,
        monitor="finetune-validate/loss",
    )

    trainer = Trainer(
        logger=WandbLogger(project="windmark", name=label),
        precision="bf16-mixed",
        gradient_clip_val=params.gradient_clip_val,
        max_epochs=params.max_finetune_epochs,
        min_epochs=params.n_epochs_frozen,
        callbacks=[
            RichProgressBar(),
            EarlyStopping(monitor="finetune-validate/loss", patience=params.patience),
            ThawedFinetuning(transition=params.n_epochs_frozen),
            LearningRateMonitor(logging_interval="step"),
            checkpointer,
        ],
    )

    trainer.fit(module)

    module = SequenceModule.load_from_checkpoint(
        checkpoint_path=str(checkpointer.best_model_path),
        datapath=str(lifestreams.path),
        params=params,
        manager=manager,
        mode="finetune",
    )

    # trainer.test(module)

    return fl.FlyteFile(checkpointer.best_model_path)
