import polars as pl

from windmark.core.managers import SchemaManager
from windmark.core.orchestration import task


@task
def diagnose(ledger: pl.DataFrame, schema: SchemaManager):
    assert schema.sequence_id in ledger.columns
    assert schema.event_id in ledger.columns
    assert schema.order_by in ledger.columns
    assert schema.target_id in ledger.columns

    for field in schema.fields:
        assert field.name in ledger.columns

    assert any(
        ledger.get_column(schema.sequence_id).dtype.is_numeric(),
        ledger.get_column(schema.sequence_id).dtype.is_string(),
    )

    assert any(
        ledger.get_column(schema.event_id).dtype.is_numeric(),
        ledger.get_column(schema.event_id).dtype.is_string(),
    )

    assert ledger.get_column(schema.event_order).dtype.is_numeric()
    assert ledger.get_column(schema.target_id).dtype.is_string()

    assert ledger.get_column(schema.sequence_id).null_count() == 0
    assert ledger.get_column(schema.event_id).null_count() == 0
    assert ledger.get_column(schema.event_order).null_count() == 0
