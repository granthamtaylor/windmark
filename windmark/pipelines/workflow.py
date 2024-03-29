from functools import partial

import flytekit as fk

from windmark.core.structs import Hyperparameters
from windmark.core.managers import SchemaManager, SplitManager
import windmark.components as components


@fk.workflow
def pipeline(
    ledger_path: str,
    schema: SchemaManager,
    params: Hyperparameters,
    split: SplitManager,
):
    ledger = components.sanitize(ledger=ledger_path)

    fields = components.fan.fields(schema=schema)

    parsed_fields = fk.map_task(partial(components.parse, ledger=ledger))(field=fields)

    parsed_schema = components.collect.fields(schema=schema, fields=parsed_fields)

    fanned_centroids = fk.map_task(partial(components.digest, ledger=ledger, slice_size=10_000))(field=fields)

    centroids = components.collect.centroids(centroids=fanned_centroids)

    task = components.task(ledger=ledger, schema=parsed_schema, params=params)

    sample = components.sample(ledger=ledger, params=params, task=task, split=split)

    manager = components.manager(schema=parsed_schema, task=task, sample=sample, split=split, centroids=centroids)

    lifestreams = components.preprocess(ledger=ledger, manager=manager, slice_size=10)

    components.fit(lifestreams=lifestreams, params=params, manager=manager)
