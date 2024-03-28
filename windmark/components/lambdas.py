from types import SimpleNamespace

import flytekit as fk

from windmark.core.structs import Field, Centroid
from windmark.core.managers import SchemaManager, CentroidManager


@fk.task
def fan_fields(schema: SchemaManager) -> list[Field]:
    return schema.fields


@fk.task
def collect_fields(schema: SchemaManager, fields: list[Field]) -> SchemaManager:
    schema.fields = fields

    return schema


@fk.task
def collect_centroids(centroids: list[Centroid]) -> CentroidManager:
    return CentroidManager(centroids=centroids)


fan = SimpleNamespace(
    fields=fan_fields,
)

collect = SimpleNamespace(
    fields=collect_fields,
    centroids=collect_centroids,
)
