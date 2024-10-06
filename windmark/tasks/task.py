from pathlib import Path
from collections import Counter
from functools import reduce

from flytekit.types import directory

from windmark.core.managers import SchemaManager, SupervisedTaskManager, BalanceManager
from windmark.core.constructs.general import Hyperparameters
from windmark.core.orchestration import task
from windmark.core.processors import multithread, count


@task
def create_task_manager(
    lifestreams: directory.FlyteDirectory, schema: SchemaManager, params: Hyperparameters
) -> SupervisedTaskManager:
    """
    Creates a task manager for supervised machine learning tasks.

    Args:
        lifestreams (directory.FlyteDirectory): The directory containing the lifestreams.
        schema (SchemaManager): The schema manager for the task.
        params (Hyperparameters): model hyperparameters.

    Returns:
        SupervisedTaskManager: The created task manager.
    """

    path = Path(lifestreams.path)

    results: list[Counter] = multithread(process=count, key=schema.target_id, path=path)

    counter = reduce(lambda a, b: a + b, results)

    targets: dict[str, int] = dict(counter)

    if None in targets.keys():
        unlabeled: int = targets.pop(None)
    else:
        unlabeled: int = 0

    labels: list[str] = list(targets.keys())
    counts: list[int] = list(targets.values())

    balancer = BalanceManager(labels=labels, counts=counts, kappa=params.interpolation_rate, unlabeled=unlabeled)

    return SupervisedTaskManager(task="classification", n_targets=len(targets), balancer=balancer)
