from types import SimpleNamespace

import flytekit as fk

from windmark.core.constructs import Field, Centroid, LevelSet, Hyperparameters
from windmark.core.managers import SchemaManager, CentroidManager, LevelManager


@fk.task(cache=True, cache_version="1.0")
def fan_fields(schema: SchemaManager) -> list[Field]:
    return schema.fields


fan = SimpleNamespace(
    fields=fan_fields,
)


@fk.task(cache=True, cache_version="1.0")
def collect_centroids(centroids: list[Centroid]) -> CentroidManager:
    return CentroidManager(centroids=centroids)


@fk.task(cache=True, cache_version="1.0")
def collect_levelsets(levelsets: list[LevelSet]) -> LevelManager:
    return LevelManager(levelsets=levelsets)


collect = SimpleNamespace(
    centroids=collect_centroids,
    levelsets=collect_levelsets,
)


@fk.task(cache=True, cache_version="1.0")
def extract_kappa(params: Hyperparameters) -> float:
    return params.interpolation_rate


@fk.task(cache=True, cache_version="1.0")
def extract_batch_size(params: Hyperparameters) -> int:
    return params.batch_size


@fk.task(cache=True, cache_version="1.0")
def extract_n_pretrain_steps(params: Hyperparameters) -> int:
    return params.n_pretrain_steps


@fk.task(cache=True, cache_version="1.0")
def extract_n_finetune_steps(params: Hyperparameters) -> int:
    return params.n_finetune_steps


extract = SimpleNamespace(
    kappa=extract_kappa,
    batch_size=extract_batch_size,
    n_pretrain_steps=extract_n_pretrain_steps,
    n_finetune_steps=extract_n_finetune_steps,
)
