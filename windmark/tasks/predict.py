from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers.wandb import WandbLogger

import flytekit as fk
from flytekit.types import file, directory

from windmark.core.architecture.encoders import SequenceModule
from windmark.core.callbacks import ParquetBatchWriter
from windmark.core.managers import SystemManager
from windmark.core.constructs.general import Hyperparameters
from windmark.core.orchestration import task


@task
def predict_sequence_encoder(
    checkpoint: file.FlyteFile,
    lifestreams: directory.FlyteDirectory,
    params: Hyperparameters,
    manager: SystemManager,
    label: str,
) -> file.FlyteFile:
    """
    Predicts the sequence using an encoder model.

    Args:
        checkpoint (file.FlyteFile): The checkpoint file containing the finetuned model.
        lifestreams (directory.FlyteDirectory): The directory containing the input data.
        params (Hyperparameters): The hyperparameters for the model.
        manager (SystemManager): The system state manager.
        label (str): The name for the experiment.

    Returns:
        file.FlyteFile: Model score predictions
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

    outpath = Path(fk.current_context().working_directory) / "predictions.parquet"

    trainer = Trainer(
        logger=WandbLogger(name="windmark", version=label),
        precision="bf16-mixed",
        callbacks=[
            RichProgressBar(),
            ParquetBatchWriter(str(outpath)),
        ],
    )

    print(f"writing predictions to '{outpath}'")

    trainer.predict(module, return_predictions=False)

    return file.FlyteFile(str(outpath))
