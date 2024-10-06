from types import SimpleNamespace


from windmark.core.constructs.general import Centroid, LevelSet, Hyperparameters, FieldRequest
from windmark.core.managers import SchemaManager, CentroidManager, LevelManager
from windmark.core.orchestration import task


@task
def fan_fields(schema: SchemaManager) -> list[FieldRequest]:
    """
    Retrieves the fields from the given schema.

    Args:
        schema (SchemaManager): The schema from which to retrieve the fields.

    Returns:
        list[FieldRequest]: The list of fields from the schema.
    """
    return schema.fields


fan = SimpleNamespace(
    fields=fan_fields,
)


@task
def collect_centroids(centroids: list[Centroid]) -> CentroidManager:
    """
    Collects a list of centroids and returns a CentroidManager object.

    Args:
        centroids (list[Centroid]): A list of centroids.

    Returns:
        CentroidManager: A CentroidManager object.

    """
    return CentroidManager(centroids=centroids)


@task
def collect_levelsets(levelsets: list[LevelSet]) -> LevelManager:
    """
    Collects a list of LevelSet objects and returns a LevelManager object.

    Args:
        levelsets (list[LevelSet]): A list of LevelSet objects.

    Returns:
        LevelManager: A LevelManager object containing the collected levelsets.
    """
    return LevelManager(levelsets=levelsets)


collect = SimpleNamespace(
    centroids=collect_centroids,
    levelsets=collect_levelsets,
)


@task
def extract_kappa(params: Hyperparameters) -> float:
    """
    Extracts the interpolation rate (kappa) from the given Hyperparameters object.

    Parameters:
        params (Hyperparameters): The Hyperparameters object containing the interpolation rate.

    Returns:
        float: The interpolation rate (kappa).
    """
    return params.interpolation_rate


@task
def extract_batch_size(params: Hyperparameters) -> int:
    """
    Extracts the batch size from the given Hyperparameters object.

    Args:
        params (Hyperparameters): The Hyperparameters object containing the batch size.

    Returns:
        int: The batch size value.
    """
    return params.batch_size


@task
def extract_n_pretrain_steps(params: Hyperparameters) -> int:
    """
    Extracts the number of pretrain steps from the given Hyperparameters object.

    Args:
        params (Hyperparameters): The Hyperparameters object containing the pretrain steps.

    Returns:
        int: The number of pretrain steps.
    """
    return params.n_pretrain_steps


@task
def extract_n_finetune_steps(params: Hyperparameters) -> int:
    """
    Extracts the number of finetune steps from the given Hyperparameters object.

    Args:
        params (Hyperparameters): The Hyperparameters object containing the finetune steps.

    Returns:
        int: The number of finetune steps.
    """
    return params.n_finetune_steps


extract = SimpleNamespace(
    kappa=extract_kappa,
    batch_size=extract_batch_size,
    n_pretrain_steps=extract_n_pretrain_steps,
    n_finetune_steps=extract_n_finetune_steps,
)
