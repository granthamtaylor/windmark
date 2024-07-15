from pathlib import Path
from collections import Counter
from functools import reduce

from flytekit.types import directory

from windmark.core.managers import SchemaManager, SplitManager
from windmark.core.orchestration import task
from windmark.components.processors import multithread, count


@task
def create_split_manager(
    lifestreams: directory.FlyteDirectory,
    schema: SchemaManager,
) -> SplitManager:
    path = Path(lifestreams.path)

    results: list[Counter] = multithread(n_workers=16, process=count, key=schema.split_id, path=path)

    counter = reduce(lambda a, b: a.update(b), results)

    splits: dict[str, int] = dict(counter)

    for split in list(splits.keys()):
        assert split in ["train", "test", "validate"]

    return SplitManager(**splits)
