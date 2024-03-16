from flytekit import task
import polars as pl

from source.core import LabelBalancer, Hyperparameters

@task
def rebalance_class_labels(
    ledger: pl.DataFrame,
    params: Hyperparameters,
) -> LabelBalancer:

    assert "target" in ledger.columns

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
    
    balancer.interpolate(kappa=params.interpolation_rate)
    
    return balancer
