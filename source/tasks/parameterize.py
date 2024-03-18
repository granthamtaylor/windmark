import flytekit as fk

from source.core.schema import Field, Hyperparameters


@fk.task
def create_hyperparameters(fields: list[Field], params: dict[str, float] = {}) -> Hyperparameters:

    return Hyperparameters(fields=fields, **params)
