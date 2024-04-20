from types import SimpleNamespace


from windmark.core.constructs.general import Centroid, LevelSet, Hyperparameters, FieldRequest
from windmark.core.managers import SchemaManager, CentroidManager, LevelManager
from windmark.core.orchestration import task


@task
def fan_fields(schema: SchemaManager) -> list[FieldRequest]:
    return schema.fields


fan = SimpleNamespace(
    fields=fan_fields,
)


@task
def collect_centroids(centroids: list[Centroid]) -> CentroidManager:
    return CentroidManager(centroids=centroids)


@task
def collect_levelsets(levelsets: list[LevelSet]) -> LevelManager:
    return LevelManager(levelsets=levelsets)


collect = SimpleNamespace(
    centroids=collect_centroids,
    levelsets=collect_levelsets,
)


@task
def extract_kappa(params: Hyperparameters) -> float:
    return params.interpolation_rate


@task
def extract_batch_size(params: Hyperparameters) -> int:
    return params.batch_size


@task
def extract_n_pretrain_steps(params: Hyperparameters) -> int:
    return params.n_pretrain_steps


@task
def extract_n_finetune_steps(params: Hyperparameters) -> int:
    return params.n_finetune_steps


extract = SimpleNamespace(
    kappa=extract_kappa,
    batch_size=extract_batch_size,
    n_pretrain_steps=extract_n_pretrain_steps,
    n_finetune_steps=extract_n_finetune_steps,
)
