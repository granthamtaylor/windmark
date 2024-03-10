from flytekit import task
import polars as pl

from source.core import LabelBalancer

@task
def rebalance_class_labels(
    ledger: pl.DataFrame,
    kappa: float,
) -> LabelBalancer:
    assert "target" in ledger.columns

    assert 0.0 <= kappa <= 1.0

    records: dict[str, list[float]] = (
        ledger.get_column("target")
        .value_counts()
        .select(
            labels=pl.col("target"),
            counts=pl.col("count")
        )
        .to_dict(as_series=False)
    )

    balancer = LabelBalancer(
        labels=records['labels'],
        counts=records['counts'],
    )
    
    balancer.interpolate(kappa=kappa)
    
    return balancer
