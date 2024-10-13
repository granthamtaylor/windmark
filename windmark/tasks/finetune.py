import flytekit as fk
from flytekit.types import directory, file
from datetime import datetime

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar, LearningRateMonitor
from lightning.pytorch.loggers.wandb import WandbLogger

from windmark.core.architecture.encoders import SequenceModule
from windmark.core.managers import SystemManager
from windmark.core.constructs.general import Hyperparameters
from windmark.core.callbacks import ThawedFinetuning
from windmark.core.orchestration import task


@task(requests=fk.Resources(cpu="32", mem="64Gi"), cache_ignore_input_vars=tuple(["label"]))
def finetune_sequence_encoder(
    lifestreams: directory.FlyteDirectory,
    checkpoint: file.FlyteFile,
    params: Hyperparameters,
    manager: SystemManager,
    label: str,
) -> file.FlyteFile:
    """
    Finetunes a pretrained sequence encoder model using the provided lifestreams data, checkpoint, hyperparameters, and system manager.

    Args:
        lifestreams (directory.FlyteDirectory): The directory containing the lifestreams data.
        checkpoint (file.FlyteFile): The pretrained checkpoint file to load the initial model weights from.
        params (Hyperparameters): The hyperparameters for the finetuning process.
        manager (SystemManager): The system state manager.
        label (str): The name for the experiment.

    Returns:
        file.FlyteFile: The file object representing the best model checkpoint after finetuning.
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

    timestamp = datetime.now().strftime("%Y-%m-%d|%H:%M")

    checkpointer = ModelCheckpoint(
        dirpath=fk.current_context().working_directory,
        monitor="finetune-validate/loss",
        filename=f"{label}:{timestamp}",
    )

    trainer = Trainer(
        logger=WandbLogger(name="windmark", version=f"{label}:{timestamp}"),
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

    return file.FlyteFile(checkpointer.best_model_path)
