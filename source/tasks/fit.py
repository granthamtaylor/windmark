from pathlib import Path

from funcy import join
import flytekit as fk
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateFinder, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np

from source.core.schema import Hyperparameters, Field
from source.core.architecture import SequenceModule
from source.core.utils import LabelBalancer

@fk.task(requests=fk.Resources(cpu="24", mem="8Gi"))
def fit_sequence_encoder(
    dataset: fk.types.directory.FlyteDirectory,
    params: Hyperparameters,
    fields: list[Field],
    centroids: list[dict[str, np.ndarray]],
    balancer: LabelBalancer,
) -> SequenceModule:
    
    assert torch.cuda.is_available()
    
    torch.set_float32_matmul_precision('medium')

    module = SequenceModule(
        datapath=dataset.path,
        fields=fields,
        params=params,
        centroids=join(centroids),
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
        max_epochs=params.max_epochs,
        gradient_clip_val=params.gradient_clip_val,
        fast_dev_run=params.dev_mode,
        callbacks = [
            LearningRateFinder(),
            EarlyStopping(monitor='pretrain-validate/loss'),
            pretrain := ModelCheckpoint(),
            
        ]
    )

    trainer.fit(module)
    trainer.test(module)

    module.mode = 'finetune'
    trainer = Trainer(
        logger=logger,
        default_root_dir=root/'finetune',
        accelerator="auto",
        devices="auto",
        strategy="auto",
        precision="bf16-mixed",
        max_epochs=params.max_epochs,
        gradient_clip_val=params.gradient_clip_val,
        fast_dev_run=params.dev_mode,
        callbacks=[
            LearningRateFinder(),
            EarlyStopping(monitor='finetune-validate/loss'),
            finetune := ModelCheckpoint(),
        ]
    )

    trainer.fit(module)
    trainer.test(module)

    return module
