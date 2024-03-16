from pathlib import Path

import flytekit as fk
from funcy import join
import torch
from tdigest import TDigest
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    ModelSummary,
    LearningRateFinder,
    DeviceStatsMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger

from source.core import Hyperparameters, SequenceModule, LabelBalancer


@fk.task(requests=fk.Resources(cpu="24", mem="8Gi"))
def train_sequence_encoder(
    dataset: fk.types.directory.FlyteDirectory,
    params: Hyperparameters,
    digests: list[dict[str, TDigest]],
    balancer: LabelBalancer,
) -> SequenceModule:
    
    assert torch.cuda.is_available()

    module = SequenceModule(
        datapath=dataset.path,
        params=params,
        digests=join(digests),
        balancer=balancer,
    )
    
    root = Path(fk.current_context().working_directory) / "checkpoints"
    root.mkdir()
    
    logger = TensorBoardLogger("logs", name="windmark")

    trainer = Trainer(
        logger=logger,
        default_root_dir=root/'pretrain',
        accelerator="auto",
        devices="auto",
        strategy="auto",
        precision="bf16-mixed",
        max_epochs=params.pretrain.max_epochs,
        check_val_every_n_epoch=params.pretrain.check_val_every_n_epoch,
        gradient_clip_val=params.pretrain.gradient_clip_val,
        fast_dev_run=params.dev_mode,
        callbacks = [
            LearningRateFinder(),
            DeviceStatsMonitor(),
            EarlyStopping(monitor='pretrain-validate/loss'),
            ModelSummary(4),
            pretrain := ModelCheckpoint(),
            
        ]
    )

    trainer.fit(module)

    module.mode = 'finetune'
    trainer = Trainer(
        logger=logger,
        default_root_dir=root/'finetune',
        accelerator="auto",
        devices="auto",
        strategy="auto",
        precision="bf16-mixed",
        max_epochs=params.finetune.max_epochs,
        check_val_every_n_epoch=params.finetune.check_val_every_n_epoch,
        gradient_clip_val=params.finetune.gradient_clip_val,
        fast_dev_run=params.dev_mode,
        callbacks=[
            LearningRateFinder(),
            DeviceStatsMonitor(),
            EarlyStopping(monitor='finetune-validate/loss'),
            ModelSummary(4),
            finetune := ModelCheckpoint(),
        ]
    )

    trainer.fit(module)

    return module
