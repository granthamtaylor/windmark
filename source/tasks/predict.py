from pathlib import Path

import flytekit as fl
from lightning.pytorch import Trainer

from source.core import SequenceModule, ParquetBatchWriter


@fl.task
def predict_sequence_encoder(module: SequenceModule) -> fl.types.file.FlyteFile:
    outpath = Path(fl.current_context().working_directory) / "lifestreams"

    callbacks = [ParquetBatchWriter(outpath)]

    trainer = Trainer(accelerator="cpu", fast_dev_run=True, callbacks=callbacks)

    trainer.predict(module)

    return fl.types.file.FlyteFile(outpath)
