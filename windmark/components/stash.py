from flytekit.types import file

from windmark.core.constructs.general import Hyperparameters
from windmark.core.managers import SystemManager, LabelManager
from windmark.core.orchestration import task


@task
def pretrain_sequence_encoder(model: file.FlyteFile, manager: SystemManager, params: Hyperparameters):
    _: str = LabelManager.from_path(model.path, add_date=False)
