from pathlib import Path
import json

from flytekit.types import file

from windmark.core.constructs.general import Hyperparameters
from windmark.core.managers import SystemManager, LabelManager
from windmark.core.orchestration import task


@task
def stash_model_state(checkpoint: file.FlyteFile, manager: SystemManager, params: Hyperparameters):
    """
    Stash state required to reproduce a model.

    Args:
        model (file.FlyteFile): The model file to pretrain.
        manager (SystemManager): The system state manager.
        params (Hyperparameters): The hyperparameters for pretraining.

    Returns:
        directory.FlyteDirectory
    """

    version = LabelManager.inference(checkpoint.path)

    path = Path("./model") / version

    with open(path / "manager.json") as file:
        json.dump(manager.to_json(), file)

    with open(path / "params.json") as file:
        json.dump(params.to_json(), file)
