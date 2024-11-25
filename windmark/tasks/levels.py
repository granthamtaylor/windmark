# Copyright Grantham Taylor.

from pathlib import Path
from collections import Counter
from functools import reduce

import flytekit as fl

from windmark.core.constructs.managers import LevelSet
from windmark.core.dev.interface import FieldRequest, FieldType
from windmark.core.data.processors import multithread, count
from windmark.orchestration.environments import context


@context.default
def create_unique_levels_from_lifestream(lifestreams: fl.FlyteDirectory, field: FieldRequest) -> LevelSet | None:
    """
    Create unique levels from a lifestream.

    Args:
        lifestreams (fl.FlyteDirectory): The lifestream directory.
        field (FieldRequest): The field to create levels for.

    Returns:
        LevelSet|None: The set of unique levels for the given field.
    """

    if field.type not in [FieldType.Category, FieldType.Categories]:
        return None

    print(f'- creating state manager for field "{field.name}"')

    path = Path(lifestreams.path)

    results: list[Counter] = multithread(process=count, key=field.name, path=path)

    counter = reduce(lambda a, b: a + b, results)

    levels = dict(counter)

    if None in levels.keys():
        del levels[None]

    return LevelSet(name=field.name, levels=list(levels.keys()))
