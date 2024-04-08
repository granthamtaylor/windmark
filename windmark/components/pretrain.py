from pathlib import Path

import flytekit as fk

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from windmark.core.architecture import SequenceModule
from windmark.core.managers import SystemManager
from windmark.core.structs import Hyperparameters


@fk.task(requests=fk.Resources(cpu="32", mem="64Gi"))
def pretrain_sequence_encoder(
    lifestreams: fk.types.directory.FlyteDirectory,
    params: Hyperparameters,
    manager: SystemManager,
) -> fk.types.file.FlyteFile:
    assert torch.cuda.is_available()

    torch.set_float32_matmul_precision("medium")

    module = SequenceModule(
        datapath=lifestreams.path,
        params=params,
        manager=manager,
        mode="pretrain",
    )

    root = Path(fk.current_context().working_directory) / "checkpoints"
    root.mkdir()

    trainer = Trainer(
        logger=TensorBoardLogger("logs", name="windmark", version=manager.version),
        accelerator="auto",
        devices="auto",
        strategy="auto",
        precision="bf16-mixed",
        gradient_clip_val=params.gradient_clip_val,
        max_epochs=params.max_epochs,
        default_root_dir=root / "pretrain",
        callbacks=[
            RichProgressBar(),
            EarlyStopping(monitor="pretrain-validate/loss", patience=params.patience),
            checkpoint := ModelCheckpoint(root / "pretrain"),
        ],
    )

    trainer.fit(module)
    trainer.test(module)

    return fk.types.file.FlyteFile(checkpoint.best_model_path)
