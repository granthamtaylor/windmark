from functools import partial

import flytekit as fk

from windmark.core.constructs.general import Hyperparameters
from windmark.core.managers import SchemaManager, SplitManager
import windmark.components as lib


@fk.workflow
def train(datapath: str, schema: SchemaManager, params: Hyperparameters, split: SplitManager):
    ledger = lib.sanitize(ledger=datapath)

    kappa = lib.extract.kappa(params=params)

    batch_size = lib.extract.batch_size(params=params)

    n_pretrain_steps = lib.extract.n_pretrain_steps(params=params)

    n_finetune_steps = lib.extract.n_finetune_steps(params=params)

    fields = lib.fan.fields(schema=schema)

    fk.map_task(partial(lib.parse, ledger=ledger))(field=fields)

    fanned_centroids = fk.map_task(partial(lib.digest, ledger=ledger, slice_size=10_000))(field=fields)

    centroids = lib.collect.centroids(centroids=fanned_centroids)

    fanned_levelsets = fk.map_task(partial(lib.levels, ledger=ledger))(field=fields)

    levelsets = lib.collect.levelsets(levelsets=fanned_levelsets)

    task = lib.task(ledger=ledger, schema=schema, kappa=kappa)

    sample = lib.sample(
        ledger=ledger,
        batch_size=batch_size,
        n_pretrain_steps=n_pretrain_steps,
        n_finetune_steps=n_finetune_steps,
        task=task,
        split=split,
    )

    system = lib.system(schema=schema, task=task, sample=sample, split=split, centroids=centroids, levelsets=levelsets)

    lifestreams = lib.preprocess(ledger=ledger, manager=system, slice_size=10)

    pretrained = lib.pretrain(lifestreams=lifestreams, params=params, manager=system)

    # pretrained = "checkpoints/pretrain/woods-hill:DCQO.ckpt"

    finetuned = lib.finetune(checkpoint=pretrained, lifestreams=lifestreams, params=params, manager=system)

    lib.predict(checkpoint=finetuned, params=params, manager=system, lifestreams=lifestreams)
