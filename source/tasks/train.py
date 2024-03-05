import flytekit as fl
from funcy import join
from tdigest import TDigest
from lightning.pytorch import Trainer

from source.core import Hyperparameters, SequenceModule

@fl.task
def train_sequence_encoder(
    dataset: fl.types.directory.FlyteDirectory,
    params: Hyperparameters,
    digests: list[dict[str, TDigest]],
) -> SequenceModule:

    module = SequenceModule(
        datapath=dataset.path,
        params=params,
        digests=join(digests),
    )

    trainer = Trainer(accelerator="cpu")
    trainer.fit(module)

    module.mode = 'finetune'
    trainer = Trainer(accelerator="cpu")
    trainer.fit(module)

    return module
