import flytekit as fk
import polars as pl

from windmark.core.constructs import Field
from windmark.core.orchestration import task


@task(requests=fk.Resources(cpu="32", mem="64Gi"))
def parse_field_from_ledger(
    ledger: str,
    field: Field,
) -> Field:
    lf = pl.scan_parquet(ledger)

    assert field.name in lf.columns, f"column {field.name} is not available"

    return field
