from pathlib import Path
from collections import Counter
from functools import reduce

from flytekit.types import directory

from windmark.core.managers import LevelSet
from windmark.core.constructs.interface import FieldRequest, FieldType
from windmark.core.orchestration import task
from windmark.components.processors import multithread, count


@task
def create_unique_levels_from_lifestream(lifestreams: directory.FlyteDirectory, field: FieldRequest) -> LevelSet:
    if field.type not in [FieldType.Category, FieldType.Categories]:
        return LevelSet.empty(name=field.name)

    print(f'- creating state manager for field "{field.name}"')

    path = Path(lifestreams.path)

    results: list[Counter] = multithread(n_workers=16, process=count, key=field.name, path=path)

    counter = reduce(lambda a, b: a.update(b), results)

    levels = dict(counter)

    if None in levels.keys():
        del levels[None]

    return LevelSet.from_levels(name=field.name, levels=(levels.keys()))
