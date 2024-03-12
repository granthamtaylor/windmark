from flytekit import task
import polars as pl

from source.core.schema import Field


@task
def parse_field_from_ledger(ledger: pl.DataFrame, fieldreq: dict[str, str]) -> Field:
    name = fieldreq["name"]
    dtype = fieldreq["dtype"]

    assert name in ledger.columns, f'column {name} is not available'
    assert dtype in ["discrete", "continuous", "entity"]

    if dtype == "discrete":
        n_levels = ledger.get_column(name).n_unique()

    else:
        n_levels = None

    return Field(name, dtype, n_levels)
