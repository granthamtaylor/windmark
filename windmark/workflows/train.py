from functools import partial

import flytekit as fk

from windmark.core.constructs.general import Hyperparameters
from windmark.core.managers import SchemaManager
import windmark.tasks as lib


@fk.workflow
def train(lifestreams: fk.FlyteDirectory, schema: SchemaManager, params: Hyperparameters) -> fk.FlyteFile:
    """
    Trains a model using the provided data and hyperparameters.

    Args:
        datapath (fk.FlyteDirectory): The path to the data.
        schema (SchemaManager): The schema manager object.
        params (Hyperparameters): The hyperparameters object.
    """

    label = lib.label()

    fk.map_task(partial(lib.compare, lifestreams=lifestreams, schema=schema))(field=schema.fields)

    split = lib.split(lifestreams=lifestreams, schema=schema)

    task = lib.task(lifestreams=lifestreams, schema=schema, interpolation_rate=params.interpolation_rate)

    sample = lib.sample(task=task, split=split, params=params)

    centroids = fk.map_task(partial(lib.digest, lifestreams=lifestreams))(field=schema.fields)

    levelsets = fk.map_task(partial(lib.levels, lifestreams=lifestreams))(field=schema.fields)

    system = lib.system(schema=schema, task=task, sample=sample, split=split, centroids=centroids, levelsets=levelsets)

    pretrained = lib.pretrain(lifestreams=lifestreams, params=params, manager=system, label=label)

    finetuned = lib.finetune(checkpoint=pretrained, lifestreams=lifestreams, params=params, manager=system, label=label)

    predictions = lib.predict(checkpoint=finetuned, params=params, manager=system, lifestreams=lifestreams, label=label)

    return predictions
