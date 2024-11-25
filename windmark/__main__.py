# Copyright Grantham Taylor.

import hydra
import flytekit as fl

from windmark.core.constructs.general import Hyperparameters
from windmark.core.constructs.managers import SchemaManager
from windmark.workflows.train import train


@hydra.main(version_base=None, config_path="config", config_name="config")
def windmark(config) -> None:
    train(
        lifestreams=fl.FlyteDirectory(config.data.path),
        schema=SchemaManager.new(**config.data.structure),
        params=Hyperparameters(**config.model),
    )


if __name__ == "__main__":
    windmark()
