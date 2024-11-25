# Copyright Grantham Taylor.

from pathlib import Path
import json

import flytekit as fl

from windmark.core.constructs.general import Hyperparameters
from windmark.core.constructs.managers import SystemManager
from windmark.orchestration.environments import context


@context.default
def stash_model_state(checkpoint: fl.FlyteFile, manager: SystemManager, params: Hyperparameters, label: str):
    """
    Stash state required to reproduce a model.

    Args:
        model (fl.FlyteFile): The model file to pretrain.
        manager (SystemManager): The system state manager.
        params (Hyperparameters): The hyperparameters for pretraining.
        label (str): The name for the experiment.

    Returns:
        fl.FlyteDirectory
    """

    path = Path("./model") / label

    with open(path / "manager.json") as file:
        json.dump(manager.to_json(), file)

    with open(path / "params.json") as file:
        json.dump(params.to_json(), file)
