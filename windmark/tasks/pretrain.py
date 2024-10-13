import flytekit as fk
from flytekit.types import file, directory
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
    LearningRateMonitor,
)
from lightning.pytorch.loggers.wandb import WandbLogger

from windmark.core.architecture.encoders import SequenceModule
from windmark.core.managers import SystemManager
from windmark.core.constructs.general import Hyperparameters
from windmark.core.orchestration import task


@task(requests=fk.Resources(cpu="32", mem="64Gi"), cache_ignore_input_vars=tuple(["label"]))
def pretrain_sequence_encoder(
    lifestreams: directory.FlyteDirectory,
    params: Hyperparameters,
    manager: SystemManager,
    label: str,
) -> file.FlyteFile:
    """
    Pretrains a sequence encoder model using the provided lifestreams, hyperparameters, and system manager.

    Args:
        lifestreams (directory.FlyteDirectory): The directory containing the lifestreams data.
        params (Hyperparameters): The hyperparameters for pretraining.
        manager (SystemManager): The system state manager.
        label (str): The name for the experiment.

    Returns:
        file.FlyteFile: The path to the best model checkpoint file.
    """

    torch.set_float32_matmul_precision("medium")
    torch.multiprocessing.set_sharing_strategy("file_system")

    module = SequenceModule(
        datapath=str(lifestreams.path),
        params=params,
        manager=manager,
        mode="pretrain",
    )

    checkpointer = ModelCheckpoint(
        dirpath=fk.current_context().working_directory, monitor="pretrain-total-validate/loss", filename=label
    )

    trainer = Trainer(
        logger=WandbLogger(name="windmark", version=label),
        precision="bf16-mixed",
        gradient_clip_val=params.gradient_clip_val,
        max_epochs=params.max_pretrain_epochs,
        callbacks=[
            RichProgressBar(),
            EarlyStopping(monitor="pretrain-total-validate/loss", patience=params.patience),
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
        mode="pretrain",
    )

    trainer.test(module)

    return file.FlyteFile(checkpointer.best_model_path)
