import flytekit as fk
import polars as pl

from source.core.schema import Field

@fk.task
def parse_field_from_ledger(
    ledger: str,
    fieldreq: dict[str, str],
) -> Field:
    
    lf = pl.scan_parquet(ledger)
    
    name = fieldreq["name"]
    dtype = fieldreq["dtype"]

    assert name in lf.columns, f'column {name} is not available'
    assert dtype in ["discrete", "continuous", "entity"]

    if dtype == "discrete":
        n_levels = (
            lf.select(name).collect().get_column(name).n_unique())

    else:
        n_levels = None

    return Field(name, dtype, n_levels)
