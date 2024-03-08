from flytekit import task
import polars as pl

@task
def rebalance_class_labels(
    ledger: pl.DataFrame,
    kappa: float,
) -> dict[str, float]:
    assert "target" in ledger.columns

    assert 0.0 <= kappa <= 1.0

    records: list[dict[str, str | float]] = (
        ledger.get_column("target")
        .value_counts()
        .select("target", probability=pl.col("count") / pl.sum("count"))
        .to_dicts()
    )

    targets = {record["target"]: record["probability"] for record in records}

    return targets
