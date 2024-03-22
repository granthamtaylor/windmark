from pathlib import Path

import flytekit as fk
from lightning.pytorch import Trainer

from windmark.core.architecture import SequenceModule
from windmark.core.callbacks import ParquetBatchWriter


@fk.task
def predict_sequence_encoder(module: SequenceModule) -> fk.types.file.FlyteFile:
    outpath = Path(fk.current_context().working_directory) / "lifestreams"

    callbacks = [ParquetBatchWriter(outpath)]

    trainer = Trainer(accelerator="cpu", fast_dev_run=True, callbacks=callbacks)

    trainer.predict(module)

    return fk.types.file.FlyteFile(outpath)
