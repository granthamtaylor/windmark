from flytekit import task

@task
def create_fieldreqs_from_schema(schema: dict[str, str]) -> list[dict[str, str]]:

    fieldreqs = []

    for name, dtype in schema.items():
        fieldreqs.append(dict(name=name, dtype=dtype))
    
    return fieldreqs