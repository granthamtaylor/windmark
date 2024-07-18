import flytekit as fk
from flytekit.types import directory, file

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
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
    torch.set_float32_matmul_precision("medium")
    torch.multiprocessing.set_sharing_strategy("file_system")

    version: str = LabelManager.from_path(checkpoint.path)

    module = SequenceModule.load_from_checkpoint(
        checkpoint_path=str(checkpoint.path),
        datapath=lifestreams.path,
        params=params,
        manager=manager,
        mode="finetune",
    )

    checkpointer = ModelCheckpoint(
        dirpath=f"./checkpoints/finetune/{version}",
        monitor="finetune-validate/loss",
    )
    trainer = Trainer(
        logger=TensorBoardLogger("logs", name="windmark", version=version),
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
            # StochasticWeightAveraging(swa_lrs=params.swa_lr),
            ThawedFinetuning(transition=params.n_epochs_frozen),
            checkpointer,
        ],
    )

    trainer.fit(module)
    trainer.test(module)

    return file.FlyteFile(checkpointer.best_model_path)
