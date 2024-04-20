import flytekit as fk
from flytekit.types import file, directory
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar, StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger

from windmark.core.architecture.encoders import SequenceModule
from windmark.core.managers import SystemManager, LabelManager
from windmark.core.constructs.general import Hyperparameters
from windmark.core.orchestration import task


@task(requests=fk.Resources(cpu="32", mem="64Gi"))
def pretrain_sequence_encoder(
    lifestreams: directory.FlyteDirectory,
    params: Hyperparameters,
    manager: SystemManager,
) -> file.FlyteFile:
    torch.set_float32_matmul_precision("medium")

    version: str = LabelManager.version()

    module = SequenceModule(
        datapath=str(lifestreams.path),
        params=params,
        manager=manager,
        mode="pretrain",
    )

    checkpointer = ModelCheckpoint(
        dirpath="./checkpoints/pretrain",
        monitor="pretrain-validate/loss",
        filename=version,
    )

    trainer = Trainer(
        logger=TensorBoardLogger("logs", name="windmark", version=version),
        accelerator="auto",
        devices="auto",
        strategy="auto",
        precision="bf16-mixed",
        gradient_clip_val=params.gradient_clip_val,
        max_epochs=params.max_pretrain_epochs,
        callbacks=[
            RichProgressBar(),
            EarlyStopping(monitor="pretrain-validate/loss", patience=params.patience),
            StochasticWeightAveraging(swa_lrs=params.swa_lr),
            checkpointer,
        ],
    )

    trainer.fit(module)
    trainer.test(module)

    return file.FlyteFile(checkpointer.best_model_path)
