from pathlib import Path
from collections import Counter
from functools import reduce

from flytekit.types import directory

from windmark.core.managers import SchemaManager, SplitManager
from windmark.core.orchestration import task
from windmark.core.processors import multithread, count


@task
def create_split_manager(lifestreams: directory.FlyteDirectory, schema: SchemaManager, n_workers: int) -> SplitManager:
    """
    Create a SplitManager to count number of events in each strata.

    Args:
        lifestreams (directory.FlyteDirectory): The directory containing the lifestreams.
        schema (SchemaManager): The schema manager object.
        n_workers (int): The number of workers to use for parallel processing.

    Returns:
        SplitManager: The created SplitManager object.

    Raises:
        ValueError: If any of the splits in the resulting SplitManager are not in ["train", "test", "validate"].
    """

    path = Path(lifestreams.path)

    results: list[Counter] = multithread(n_workers=n_workers, process=count, key=schema.split_id, path=path)

    counter = reduce(lambda a, b: a + b, results)

    splits: dict[str, int] = dict(counter)

    for split in list(splits.keys()):
        if split not in ["train", "test", "validate"]:
            raise ValueError("split must be one of ['train', 'test', 'validate']")

    return SplitManager(**splits)
