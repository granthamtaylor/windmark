from functools import partial

import flytekit as fk

from windmark.core.constructs.general import Hyperparameters
from windmark.core.managers import SchemaManager
import windmark.components as lib


@fk.workflow
def train(datapath: str, schema: SchemaManager, params: Hyperparameters):
    """
    Trains a model using the provided data and hyperparameters.

    Args:
        datapath (str): The path to the data.
        schema (SchemaManager): The schema manager object.
        params (Hyperparameters): The hyperparameters object.

    Returns:
        None
    """

    lifestreams = lib.sanitize(datapath=datapath)

    kappa = lib.extract.kappa(params=params)

    batch_size = lib.extract.batch_size(params=params)

    n_pretrain_steps = lib.extract.n_pretrain_steps(params=params)

    n_finetune_steps = lib.extract.n_finetune_steps(params=params)

    n_workers = lib.extract.n_workers(params=params)

    fields = lib.fan.fields(schema=schema)

    split = lib.split(lifestreams=lifestreams, schema=schema, n_workers=n_workers)

    task = lib.task(lifestreams=lifestreams, schema=schema, kappa=kappa, n_workers=n_workers)

    sample = lib.sample(
        batch_size=batch_size,
        n_pretrain_steps=n_pretrain_steps,
        n_finetune_steps=n_finetune_steps,
        task=task,
        split=split,
    )

    fanned_centroids = fk.map_task(partial(lib.digest, lifestreams=lifestreams, n_workers=n_workers))(field=fields)

    centroids = lib.collect.centroids(centroids=fanned_centroids)

    fanned_levelsets = fk.map_task(partial(lib.levels, lifestreams=lifestreams, n_workers=n_workers))(field=fields)

    levelsets = lib.collect.levelsets(levelsets=fanned_levelsets)

    system = lib.system(schema=schema, task=task, sample=sample, split=split, centroids=centroids, levelsets=levelsets)

    pretrained = lib.pretrain(lifestreams=lifestreams, params=params, manager=system)

    finetuned = lib.finetune(checkpoint=pretrained, lifestreams=lifestreams, params=params, manager=system)

    lib.predict(checkpoint=finetuned, params=params, manager=system, lifestreams=lifestreams)
