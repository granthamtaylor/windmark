from pathlib import Path

import flytekit as fk
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from windmark.core.architecture import SequenceModule
from windmark.core.callbacks import ParquetBatchWriter
from windmark.core.managers import SystemManager
from windmark.core.structs import Hyperparameters


@fk.task
def predict_sequence_encoder(
    checkpoint: fk.types.file.FlyteFile,
    lifestreams: fk.types.directory.FlyteDirectory,
    params: Hyperparameters,
    manager: SystemManager,
):
    module = SequenceModule.load_from_checkpoint(
        checkpoint_path=checkpoint.path,
        datapath=lifestreams.path,
        params=params,
        manager=manager,
        mode="inference",
    )

    outpath = Path(fk.current_context().working_directory) / "lifestreams"
    outpath.mkdir()

    trainer = Trainer(
        logger=TensorBoardLogger("logs", name="windmark"),
        accelerator="auto",
        devices="auto",
        strategy="auto",
        precision="bf16-mixed",
        callbacks=[
            RichProgressBar(),
            ParquetBatchWriter("/home/grantham/windmark/data/predictions.parquet"),
        ],
    )

    trainer.predict(module, return_predictions=False)
