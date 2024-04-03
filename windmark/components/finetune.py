from pathlib import Path

import flytekit as fk

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from windmark.core.architecture import SequenceModule
from windmark.core.callbacks import ThawedFinetuning
from windmark.core.managers import SystemManager
from windmark.core.structs import Hyperparameters


@fk.task(requests=fk.Resources(cpu="24", mem="8Gi"))
def finetune_sequence_encoder(
    lifestreams: fk.types.directory.FlyteDirectory,
    checkpoint: fk.types.file.FlyteFile,
    params: Hyperparameters,
    manager: SystemManager,
) -> fk.types.file.FlyteFile:
    assert torch.cuda.is_available()

    torch.set_float32_matmul_precision("medium")

    module = SequenceModule.load_from_checkpoint(
        checkpoint_path=checkpoint.path,
        datapath=lifestreams.path,
        params=params,
        manager=manager,
        mode="finetune",
    )

    root = Path(fk.current_context().working_directory) / "checkpoints"

    trainer = Trainer(
        logger=TensorBoardLogger("logs", name="windmark"),
        accelerator="auto",
        devices="auto",
        strategy="auto",
        precision="bf16-mixed",
        gradient_clip_val=params.gradient_clip_val,
        max_epochs=params.max_epochs,
        default_root_dir=root / "finetune",
        min_epochs=(params.n_epochs_frozen + 1),
        callbacks=[
            RichProgressBar(),
            ThawedFinetuning(transition=params.n_epochs_frozen),
            EarlyStopping(monitor="finetune-validate/loss", patience=12),
            checkpoint := ModelCheckpoint(root / "finetune"),
        ],
    )

    trainer.fit(module)
    trainer.test(module)

    return fk.types.file.FlyteFile(checkpoint.best_model_path)
