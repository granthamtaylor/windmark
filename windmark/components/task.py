import os
from collections import Counter

from flytekit.types import directory
import fastavro

from windmark.core.managers import SchemaManager, SupervisedTaskManager, BalanceManager
from windmark.core.orchestration import task


@task
def create_task_manager(
    lifestreams: directory.FlyteDirectory,
    schema: SchemaManager,
    kappa: float,
) -> SupervisedTaskManager:
    counter = Counter()

    for filename in os.listdir(lifestreams.path):
        if filename.endswith(".avro"):
            with open(f"{lifestreams.path}/{filename}", "rb") as f:
                reader = fastavro.reader(f)
                for sequence in reader:
                    counter.update(sequence[schema.target_id])

    targets: dict[str, int] = dict(counter)

    if None in targets.keys():
        unlabeled: int = targets.pop(None)
    else:
        unlabeled: int = 0

    labels: list[str] = list(targets.keys())
    counts: list[int] = list(targets.values())

    balancer = BalanceManager(labels=labels, counts=counts, kappa=kappa, unlabeled=unlabeled)

    return SupervisedTaskManager(task="classification", n_targets=len(targets), balancer=balancer)
