from pathlib import Path

import flytekit as fk
from funcy import join
from tdigest import TDigest
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    ModelSummary,
    LearningRateFinder,
    DeviceStatsMonitor,
    EarlyStopping,
    ModelCheckpoint
    # SpikeDetection,
    # OnExceptionCheckpoint,
    # ThroughputMonitor,
)

from source.core import Hyperparameters, SequenceModule

@fk.task(requests=fk.Resources(cpu="16", mem="8Gi"))
def train_sequence_encoder(
    dataset: fk.types.directory.FlyteDirectory,
    params: Hyperparameters,
    digests: list[dict[str, TDigest]],
) -> SequenceModule:

    module = SequenceModule(
        datapath=dataset.path,
        params=params,
        digests=join(digests),
    )
    
    root = Path(fk.current_context().working_directory) / "checkpoints"
    root.mkdir()

    trainer = Trainer(
        default_root_dir=root/'pretrain',
        accelerator="cpu",
        callbacks = [
            # SpikeDetection(),
            LearningRateFinder(),
            DeviceStatsMonitor(),
            EarlyStopping(monitor='pretrain-validate/loss'),
            ModelSummary(4),
            pretrain := ModelCheckpoint(),
            # ThroughputMonitor(lambda x: x[0].batch_size[0]),
            
        ]
    )

    trainer.fit(module)

    module.mode = 'finetune'
    trainer = Trainer(
        default_root_dir=root/'finetune',
        accelerator="cpu",
        callbacks=[
            # SpikeDetection(),
            LearningRateFinder(),
            DeviceStatsMonitor(),
            EarlyStopping(monitor='finetune-validate/loss'),
            ModelSummary(4),
            finetune := ModelCheckpoint(),
            # ThroughputMonitor(lambda x: x.batch_size[0]),            
        ]
    )

    trainer.fit(module)

    return module
