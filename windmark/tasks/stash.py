from pathlib import Path
import json

import flytekit as fk

from windmark.core.constructs.general import Hyperparameters
from windmark.core.managers import SystemManager
from windmark.core.orchestration import task


@task
def stash_model_state(checkpoint: fk.FlyteFile, manager: SystemManager, params: Hyperparameters, label: str):
    """
    Stash state required to reproduce a model.

    Args:
        model (fk.FlyteFile): The model file to pretrain.
        manager (SystemManager): The system state manager.
        params (Hyperparameters): The hyperparameters for pretraining.
        label (str): The name for the experiment.

    Returns:
        fk.FlyteDirectory
    """

    path = Path("./model") / label

    with open(path / "manager.json") as file:
        json.dump(manager.to_json(), file)

    with open(path / "params.json") as file:
        json.dump(params.to_json(), file)
