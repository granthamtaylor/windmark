from functools import partial

import flytekit as fk

from windmark.core.constructs import Hyperparameters
from windmark.core.managers import SchemaManager, SplitManager
import windmark.components as comp


@fk.workflow
def train(
    datapath: str,
    schema: SchemaManager,
    params: Hyperparameters,
    split: SplitManager,
):
    ledger = comp.sanitize(ledger=datapath)

    fields = comp.fan.fields(schema=schema)

    fk.map_task(partial(comp.parse, ledger=ledger))(field=fields)

    fanned_centroids = fk.map_task(partial(comp.digest, ledger=ledger, slice_size=10_000))(field=fields)

    centroids = comp.collect.centroids(centroids=fanned_centroids)

    fanned_levelsets = fk.map_task(partial(comp.levels, ledger=ledger))(field=fields)

    levelsets = comp.collect.levelsets(levelsets=fanned_levelsets)

    task = comp.task(ledger=ledger, schema=schema, params=params)

    sample = comp.sample(ledger=ledger, params=params, task=task, split=split)

    system = comp.system(schema=schema, task=task, sample=sample, split=split, centroids=centroids, levelsets=levelsets)

    lifestreams = comp.preprocess(ledger=ledger, manager=system, slice_size=10)

    pretrained = comp.pretrain(lifestreams=lifestreams, params=params, manager=system)

    finetuned = comp.finetune(checkpoint=pretrained, lifestreams=lifestreams, params=params, manager=system)

    comp.predict(checkpoint=finetuned, params=params, manager=system, lifestreams=lifestreams)
