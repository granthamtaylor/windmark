from pathlib import Path
from collections import Counter
from functools import reduce

from flytekit.types import directory

from windmark.core.managers import LevelSet
from windmark.core.constructs.interface import FieldRequest, FieldType
from windmark.core.orchestration import task
from windmark.core.processors import multithread, count


@task
def create_unique_levels_from_lifestream(lifestreams: directory.FlyteDirectory, field: FieldRequest) -> LevelSet:
    """
    Create unique levels from a lifestream.

    Args:
        lifestreams (directory.FlyteDirectory): The lifestream directory.
        field (FieldRequest): The field to create levels for.

    Returns:
        LevelSet: The set of unique levels for the given field.
    """

    if field.type not in [FieldType.Category, FieldType.Categories]:
        return LevelSet.empty(name=field.name)

    print(f'- creating state manager for field "{field.name}"')

    path = Path(lifestreams.path)

    results: list[Counter] = multithread(process=count, key=field.name, path=path)

    counter = reduce(lambda a, b: a + b, results)

    levels = dict(counter)

    if None in levels.keys():
        del levels[None]

    return LevelSet.from_levels(name=field.name, levels=(levels.keys()))
