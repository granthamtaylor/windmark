from pathlib import Path

from flytekit.types import directory

from windmark.core.managers import SchemaManager
from windmark.core.constructs.general import FieldRequest
from windmark.core.orchestration import task
from windmark.core.processors import multithread, compare

# FIXME this should be extended to include other validation checks


@task
def compare_sequence_lengths(lifestreams: directory.FlyteDirectory, schema: SchemaManager, field: FieldRequest):
    """
    Compares the lengths of the inputs and indices in the given lifestreams.

    Args:
        lifestreams (directory.FlyteDirectory): The directory containing the lifestreams.
        schema (SchemaManager): The schema manager for the task.
        field (FieldRequest): The field to compare with the index.
    """

    path = Path(lifestreams.path)

    multithread(process=compare, path=path, key=field.name, is_static=field.type.is_static, schema=schema)
