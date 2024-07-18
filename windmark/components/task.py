from pathlib import Path
from collections import Counter
from functools import reduce

from flytekit.types import directory

from windmark.core.managers import SchemaManager, SupervisedTaskManager, BalanceManager
from windmark.core.orchestration import task
from windmark.core.processors import multithread, count


@task
def create_task_manager(
    lifestreams: directory.FlyteDirectory, schema: SchemaManager, kappa: float, n_workers: int
) -> SupervisedTaskManager:
    path = Path(lifestreams.path)

    results: list[Counter] = multithread(n_workers=n_workers, process=count, key=schema.target_id, path=path)

    counter = reduce(lambda a, b: a + b, results)

    targets: dict[str, int] = dict(counter)

    if None in targets.keys():
        unlabeled: int = targets.pop(None)
    else:
        unlabeled: int = 0

    labels: list[str] = list(targets.keys())
    counts: list[int] = list(targets.values())

    balancer = BalanceManager(labels=labels, counts=counts, kappa=kappa, unlabeled=unlabeled)

    return SupervisedTaskManager(task="classification", n_targets=len(targets), balancer=balancer)
