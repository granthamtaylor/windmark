import os
from pathlib import Path

from hydra import compose, initialize
import windmark as wm

if __name__ == "__main__":
    path = os.path.relpath(Path(os.getcwd()) / "config", Path(os.path.realpath(__file__)).parent)

    with initialize(version_base=None, config_path=path):
        config = compose(config_name="config")

    schema = wm.Schema.new(**config.data.structure, **config.data.fields)

    params = wm.Hyperparameters(**config.model)

    wm.train(datapath=config.data.path, schema=schema, params=params)
