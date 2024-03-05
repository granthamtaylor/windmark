from flytekit import task
import polars as pl

from source.core.schema import Field


@task
def parse_field_from_ledger(ledger: pl.DataFrame, fieldreq: dict[str, str]) -> Field:
    name = fieldreq["name"]
    dtype = fieldreq["dtype"]

    assert name in ledger.columns
    assert dtype in ["discrete", "continuous", "entity"]

    if dtype == "discrete":
        n_levels = ledger.get_column(name).n_unique()
        if ledger.get_column(name).null_count() > 0:
            n_levels += 1

    else:
        n_levels = None

    return Field(name, dtype, n_levels)
