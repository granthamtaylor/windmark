from windmark.core.constructs.general import FieldRequest
from windmark.core.managers import SchemaManager
from windmark.core.orchestration import task


@task
def fan_out_field_requests(schema: SchemaManager) -> list[FieldRequest]:
    """
    Retrieves the fields from the given schema.

    Args:
        schema (SchemaManager): The schema from which to retrieve the fields.

    Returns:
        list[FieldRequest]: The list of fields from the schema.
    """
    return schema.fields
