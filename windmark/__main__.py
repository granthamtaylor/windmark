from omegaconf import DictConfig
import hydra

import windmark as wm


@hydra.main(version_base=None, config_path="./config", config_name="config")
def app(config: DictConfig) -> None:
    schema = wm.Schema.new(**config.data.structure, **config.data.fields)

    params = wm.Hyperparameters(**config.params)

    wm.train(datapath=config.data.path, schema=schema, params=params)


if __name__ == "__main__":
    app()
