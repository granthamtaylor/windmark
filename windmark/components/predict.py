import flytekit as fk
from flytekit.types import file, directory
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from windmark.core.architecture import SequenceModule
from windmark.core.callbacks import ParquetBatchWriter
from windmark.core.managers import SystemManager
from windmark.core.constructs import Hyperparameters


@fk.task(requests=fk.Resources(cpu="32", mem="64Gi"))
def predict_sequence_encoder(
    checkpoint: file.FlyteFile,
    lifestreams: directory.FlyteDirectory,
    params: Hyperparameters,
    manager: SystemManager,
):
    module = SequenceModule.load_from_checkpoint(
        checkpoint_path=str(checkpoint.path),
        datapath=lifestreams.path,
        params=params,
        manager=manager,
        mode="inference",
    )

    trainer = Trainer(
        logger=TensorBoardLogger("logs", name="windmark", version=manager.version),
        accelerator="auto",
        devices="auto",
        strategy="auto",
        precision="bf16-mixed",
        callbacks=[
            RichProgressBar(),
            ParquetBatchWriter(f"data/predictions/{manager.version}.parquet"),
        ],
    )

    trainer.predict(module, return_predictions=False)
