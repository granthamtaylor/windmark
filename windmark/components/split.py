from pathlib import Path
from collections import Counter
from functools import reduce

from flytekit.types import directory

from windmark.core.managers import SchemaManager, SplitManager
from windmark.core.orchestration import task
from windmark.core.processors import multithread, count


@task
def create_split_manager(lifestreams: directory.FlyteDirectory, schema: SchemaManager, n_workers: int) -> SplitManager:
    path = Path(lifestreams.path)

    results: list[Counter] = multithread(n_workers=n_workers, process=count, key=schema.split_id, path=path)

    counter = reduce(lambda a, b: a + b, results)

    splits: dict[str, int] = dict(counter)

    for split in list(splits.keys()):
        assert split in ["train", "test", "validate"]

    return SplitManager(**splits)
