# Copyright Grantham Taylor.

from pathlib import Path

import flytekit as fl

from windmark.core.constructs.managers import SchemaManager
from windmark.core.constructs.general import FieldRequest
from windmark.core.data.processors import multithread, compare
from windmark.orchestration.environments import context

# FIXME this should be extended to include other validation checks


@context.default
def compare_sequence_lengths(lifestreams: fl.FlyteDirectory, schema: SchemaManager, field: FieldRequest):
    """
    Compares the lengths of the inputs and indices in the given lifestreams.

    Args:
        lifestreams (fl.FlyteDirectory): The directory containing the lifestreams.
        schema (SchemaManager): The schema manager for the task.
        field (FieldRequest): The field to compare with the index.
    """

    path = Path(lifestreams.path)

    multithread(process=compare, path=path, key=field.name, is_static=field.type.is_static, schema=schema)
