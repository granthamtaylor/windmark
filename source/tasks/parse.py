import flytekit as fk
import polars as pl

from source.core.schema import Field

@fk.task
def parse_field_from_ledger(
    ledger: str,
    field: Field,
) -> Field:
    
    lf = pl.scan_parquet(ledger)

    assert field.name in lf.columns, f'column {field.name} is not available'

    if field.type == "discrete":

        field.levels = (
            lf.select(field.name)
            .collect()
            .get_column(field.name)
            .n_unique()
        )
    
    assert field.is_valid

    return field
