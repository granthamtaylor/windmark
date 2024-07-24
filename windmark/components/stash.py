from flytekit.types import file

from windmark.core.constructs.general import Hyperparameters
from windmark.core.managers import SystemManager, LabelManager
from windmark.core.orchestration import task


@task
def stash_model_state(model: file.FlyteFile, manager: SystemManager, params: Hyperparameters) -> None:
    """
    Stash state required to reproduce a model.

    Args:
        model (file.FlyteFile): The model file to pretrain.
        manager (SystemManager): The system state manager.
        params (Hyperparameters): The hyperparameters for pretraining.

    Returns:
        None
    """

    _: str = LabelManager.from_path(model.path, add_date=False)
