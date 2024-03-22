import flytekit as fk
import polars as pl

from windmark.core.structs import Field


@fk.task
def diagnose(ledger: pl.DataFrame, fields: list[Field]):
    assert "sequence_id" in ledger.columns
    assert "event_id" in ledger.columns
    assert "order" in ledger.columns
    # assert "strata" in ledger.columns
    assert "target" in ledger.columns
    assert "is_trigger" in ledger.columns

    for field in fields:
        assert field.name in ledger.columns

    assert any(
        ledger.get_column("sequence_id").dtype.is_numeric(),
        ledger.get_column("sequence_id").dtype.is_string(),
    )

    assert any(
        ledger.get_column("event_id").dtype.is_numeric(),
        ledger.get_column("event_id").dtype.is_string(),
    )

    assert ledger.get_column("event_order").dtype.is_numeric()
    # assert ledger.get_column("strata").dtype.is_string()
    assert ledger.get_column("is_trigger").dtype.is_boolean()
    assert ledger.get_column("target").dtype.is_string()

    assert ledger.get_column("sequence_id").null_count() == 0
    assert ledger.get_column("event_id").null_count() == 0
    assert ledger.get_column("event_order").null_count() == 0
