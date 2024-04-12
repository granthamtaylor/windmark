from pathlib import Path

import flytekit as fk
from flytekit.types import directory, file

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from windmark.core.architecture import SequenceModule
from windmark.core.managers import SystemManager
from windmark.core.constructs import Hyperparameters
from windmark.core.callbacks import ThawedFinetuning


@fk.task(requests=fk.Resources(cpu="32", mem="64Gi"))
def finetune_sequence_encoder(
    lifestreams: directory.FlyteDirectory,
    checkpoint: file.FlyteFile,
    params: Hyperparameters,
    manager: SystemManager,
) -> file.FlyteFile:
    assert torch.cuda.is_available(), "GPU not found"

    torch.set_float32_matmul_precision("medium")

    module = SequenceModule.load_from_checkpoint(
        checkpoint_path=str(checkpoint.path),
        datapath=lifestreams.path,
        params=params,
        manager=manager,
        mode="finetune",
    )

    root = Path(fk.current_context().working_directory) / "checkpoints"

    trainer = Trainer(
        logger=TensorBoardLogger("logs", name="windmark", version=manager.version),
        accelerator="auto",
        devices="auto",
        strategy="auto",
        precision="bf16-mixed",
        gradient_clip_val=params.gradient_clip_val,
        max_epochs=params.max_finetune_epochs,
        min_epochs=(params.n_epochs_frozen + 1),
        default_root_dir=root / "finetune",
        callbacks=[
            RichProgressBar(),
            EarlyStopping(monitor="finetune-validate/loss", patience=params.patience),
            # StochasticWeightAveraging(swa_lrs=params.swa_lr),
            ThawedFinetuning(transition=params.n_epochs_frozen),
            checkpoints := ModelCheckpoint(root / "finetune"),
        ],
    )

    print(f"finished finetuning (checkpoint: {checkpoints.best_model_path})")

    trainer.fit(module)
    trainer.test(module)

    return file.FlyteFile(checkpoints.best_model_path)
