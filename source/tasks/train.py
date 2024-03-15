from pathlib import Path

import flytekit as fk
from funcy import join
import torch
from tdigest import TDigest
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    ModelSummary,
    # LearningRateFinder,
    DeviceStatsMonitor,
    EarlyStopping,
    ModelCheckpoint
    # SpikeDetection,
    # OnExceptionCheckpoint,
    # ThroughputMonitor,
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
        precision="bf16",
        fast_dev_run=True,
        callbacks = [
            # SpikeDetection(),
            # LearningRateFinder(),
            DeviceStatsMonitor(),
            # EarlyStopping(monitor='pretrain-validate/loss'),
            ModelSummary(4),
            pretrain := ModelCheckpoint(),
            # ThroughputMonitor(lambda x: x[0].batch_size[0]),
            
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
        precision="bf16",
        fast_dev_run=True,
        callbacks=[
            # SpikeDetection(),
            # LearningRateFinder(),
            DeviceStatsMonitor(),
            # EarlyStopping(monitor='finetune-validate/loss'),
            ModelSummary(4),
            finetune := ModelCheckpoint(),
            # ThroughputMonitor(lambda x: x.batch_size[0]),            
        ]
    )

    trainer.fit(module)

    return module
