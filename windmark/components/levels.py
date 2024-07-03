import os

from flytekit.types import directory
import fastavro

from windmark.core.managers import LevelSet
from windmark.core.constructs.interface import FieldRequest, FieldType
from windmark.core.orchestration import task


@task
def create_unique_levels_from_lifestream(lifestreams: directory.FlyteDirectory, field: FieldRequest) -> LevelSet:
    if field.type not in [FieldType.Category, FieldType.Categories]:
        return LevelSet.empty(name=field.name)

    print(f'- creating state manager for field "{field.name}"')

    levels = set()

    for filename in os.listdir(lifestreams.path):
        if filename.endswith(".avro"):
            with open(f"{lifestreams.path}/{filename}", "rb") as f:
                reader = fastavro.reader(f)
                for sequence in reader:
                    inputs = sequence[field.name]
                    if inputs is not None:
                        levels.update(inputs)

    if None in levels:
        levels.remove(None)

    return LevelSet.from_levels(name=field.name, levels=list(levels))
