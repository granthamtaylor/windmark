from flytekit import task

from source.core.schema import Field, Hyperparameters


@task
def create_hyperparameters(fields: list[Field], params: dict = {}) -> Hyperparameters:

    params = Hyperparameters(fields=fields, **params)
    
    complexity = params.complexity(format=True)
    
    print(f"expected memory requirements: {complexity}")
    
    return params
