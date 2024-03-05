from flytekit import task

from source.core.schema import Field, Hyperparameters


@task
def create_hyperparameters(fields: list[Field], params: dict[str, int | float] = {}) -> Hyperparameters:
    return Hyperparameters(fields=fields, **params)
