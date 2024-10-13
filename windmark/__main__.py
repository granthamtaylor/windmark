import os
from pathlib import Path

from hydra import compose, initialize
from flytekit.types import directory

from windmark.core.constructs.general import Hyperparameters
from windmark.workflows.train import train
from windmark.core.managers import SchemaManager

if __name__ == "__main__":
    path = os.path.relpath(Path(os.getcwd()) / "config", Path(os.path.realpath(__file__)).parent)

    with initialize(version_base=None, config_path=path):
        config = compose(config_name="config")

    lifestreams = directory.FlyteDirectory(str(config.data.path))
    schema = SchemaManager.new(**config.data.structure, **config.data.fields)
    params = Hyperparameters(**config.model)

    train(lifestreams=lifestreams, schema=schema, params=params)
