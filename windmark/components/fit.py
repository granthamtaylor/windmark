from pathlib import Path

import flytekit as fk

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateFinder
from lightning.pytorch.loggers import TensorBoardLogger

from windmark.core.architecture import SequenceModule
from windmark.core.callbacks import ParquetBatchWriter, ThawedFinetuning
from windmark.core.managers import SequenceManager
from windmark.core.structs import Hyperparameters


@fk.task(requests=fk.Resources(cpu="24", mem="8Gi"))
def fit_sequence_encoder(
    lifestreams: fk.types.directory.FlyteDirectory,
    params: Hyperparameters,
    manager: SequenceManager,
) -> SequenceModule:
    assert torch.cuda.is_available()

    torch.set_float32_matmul_precision("medium")

    module = SequenceModule(
        datapath=lifestreams.path,
        params=params,
        manager=manager,
    )

    root = Path(fk.current_context().working_directory) / "checkpoints"
    root.mkdir()

    logger = TensorBoardLogger("logs", name="windmark")

    config = dict(
        logger=logger,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        precision="bf16-mixed",
        gradient_clip_val=params.gradient_clip_val,
        max_epochs=params.max_epochs,
    )

    trainer = Trainer(
        **config,
        default_root_dir=root / "pretrain",
        callbacks=[
            LearningRateFinder(),
            EarlyStopping(monitor="pretrain-validate/loss"),
            pretrain := ModelCheckpoint(),
        ],
    )

    print(pretrain)

    trainer.fit(module)
    trainer.test(module)

    module.mode = "finetune"

    trainer = Trainer(
        **config,
        default_root_dir=root / "finetune",
        callbacks=[
            ThawedFinetuning(transition=1),
            # LearningRateFinder(),
            EarlyStopping(monitor="finetune-validate/loss"),
            ParquetBatchWriter("/home/grantham/windmark/data/predictions.parquet"),
            finetune := ModelCheckpoint(),
        ],
    )

    print(finetune)

    trainer.fit(module)
    trainer.test(module)

    module.mode = "inference"

    trainer.predict(module)

    return module
