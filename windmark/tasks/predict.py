# Copyright Grantham Taylor.

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers.wandb import WandbLogger

import flytekit as fl

from windmark.core.architecture.encoders import SequenceModule
from windmark.core.architecture.callbacks import ParquetBatchWriter
from windmark.core.constructs.managers import SystemManager
from windmark.core.constructs.general import Hyperparameters
from windmark.orchestration.environments import context


@context.lab
def predict_sequence_encoder(
    checkpoint: fl.FlyteFile,
    lifestreams: fl.FlyteDirectory,
    params: Hyperparameters,
    manager: SystemManager,
    label: str,
) -> fl.FlyteFile:
    """
    Predicts the sequence using an encoder model.

    Args:
        checkpoint (fl.FlyteFile): The checkpoint file containing the finetuned model.
        lifestreams (fl.FlyteDirectory): The directory containing the input data.
        params (Hyperparameters): The hyperparameters for the model.
        manager (SystemManager): The system state manager.
        label (str): The name for the experiment.

    Returns:
        fl.FlyteFile: Model score predictions
    """

    torch.set_float32_matmul_precision("medium")
    torch.multiprocessing.set_sharing_strategy("file_system")

    module = SequenceModule.load_from_checkpoint(
        checkpoint_path=str(checkpoint.path),
        datapath=str(lifestreams.path),
        params=params,
        manager=manager,
        mode="inference",
    )

    out = fl.FlyteFile.new("predictions.parquet")

    trainer = Trainer(
        logger=WandbLogger(project="windmark", name=label),
        precision="bf16-mixed",
        callbacks=[
            RichProgressBar(),
            ParquetBatchWriter(out.path),
        ],
    )

    print(f"writing predictions to '{out.path}'")

    trainer.predict(module, return_predictions=False)

    return out
