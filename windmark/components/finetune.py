import flytekit as fk
from flytekit.types import directory, file

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from windmark.core.architecture.encoders import SequenceModule
from windmark.core.managers import SystemManager, LabelManager
from windmark.core.constructs.general import Hyperparameters
from windmark.core.callbacks import ThawedFinetuning
from windmark.core.orchestration import task


@task(requests=fk.Resources(cpu="32", mem="64Gi"))
def finetune_sequence_encoder(
    lifestreams: directory.FlyteDirectory,
    checkpoint: file.FlyteFile,
    params: Hyperparameters,
    manager: SystemManager,
) -> file.FlyteFile:
    """
    Finetunes a pretrained sequence encoder model using the provided lifestreams data, checkpoint, hyperparameters, and system manager.

    Args:
        lifestreams (directory.FlyteDirectory): The directory containing the lifestreams data.
        checkpoint (file.FlyteFile): The pretrained checkpoint file to load the initial model weights from.
        params (Hyperparameters): The hyperparameters for the finetuning process.
        manager (SystemManager): The system state manager.

    Returns:
        file.FlyteFile: The file object representing the best model checkpoint after finetuning.
    """

    torch.set_float32_matmul_precision("medium")
    torch.multiprocessing.set_sharing_strategy("file_system")

    version, date = LabelManager.finetune(checkpoint.path)

    module = SequenceModule.load_from_checkpoint(
        checkpoint_path=str(checkpoint.path),
        datapath=str(lifestreams.path),
        params=params,
        manager=manager,
        mode="finetune",
    )

    checkpointer = ModelCheckpoint(
        dirpath=f"./checkpoints/{version}",
        monitor="finetune-validate/loss",
        filename=f"{version}:{date}",
    )

    trainer = Trainer(
        logger=TensorBoardLogger("logs", name="windmark", version=f"{version}:{date}"),
        accelerator="auto",
        devices="auto",
        strategy="auto",
        precision="bf16-mixed",
        gradient_clip_val=params.gradient_clip_val,
        max_epochs=params.max_finetune_epochs,
        min_epochs=(params.n_epochs_frozen + 1),
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
