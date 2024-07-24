import torch
from flytekit.types import file, directory
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from windmark.core.architecture.encoders import SequenceModule
from windmark.core.callbacks import ParquetBatchWriter
from windmark.core.managers import SystemManager, LabelManager
from windmark.core.constructs.general import Hyperparameters
from windmark.core.orchestration import task


@task
def predict_sequence_encoder(
    checkpoint: file.FlyteFile,
    lifestreams: directory.FlyteDirectory,
    params: Hyperparameters,
    manager: SystemManager,
) -> file.FlyteFile:
    """
    Predicts the sequence using an encoder model.

    Args:
        checkpoint (file.FlyteFile): The checkpoint file containing the finetuned model.
        lifestreams (directory.FlyteDirectory): The directory containing the input data.
        params (Hyperparameters): The hyperparameters for the model.
        manager (SystemManager): The system state manager.

    Returns:
        file.FlyteFile: Model score predictions
    """
    torch.set_float32_matmul_precision("medium")
    torch.multiprocessing.set_sharing_strategy("file_system")

    version = LabelManager.inference(checkpoint.path)

    module = SequenceModule.load_from_checkpoint(
        checkpoint_path=str(checkpoint.path),
        datapath=str(lifestreams.path),
        params=params,
        manager=manager,
        mode="inference",
    )

    outpath = f"data/predictions/{version}.parquet"

    trainer = Trainer(
        logger=TensorBoardLogger("logs", name="windmark", version=version),
        accelerator="auto",
        devices="auto",
        strategy="auto",
        precision="bf16-mixed",
        callbacks=[
            RichProgressBar(),
            ParquetBatchWriter(outpath),
        ],
    )

    trainer.predict(module, return_predictions=False)

    return file.FlyteFile(outpath)
