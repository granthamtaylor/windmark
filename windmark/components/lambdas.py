from types import SimpleNamespace

import flytekit as fk

from windmark.core.constructs import Field, Centroid, LevelSet
from windmark.core.managers import SchemaManager, CentroidManager, LevelManager


@fk.task
def fan_fields(schema: SchemaManager) -> list[Field]:
    return schema.fields


@fk.task
def collect_centroids(centroids: list[Centroid]) -> CentroidManager:
    return CentroidManager(centroids=centroids)


@fk.task
def collect_levelsets(levelsets: list[LevelSet]) -> LevelManager:
    return LevelManager(levelsets=levelsets)


fan = SimpleNamespace(
    fields=fan_fields,
)

collect = SimpleNamespace(
    centroids=collect_centroids,
    levelsets=collect_levelsets,
)
