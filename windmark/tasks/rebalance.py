import flytekit as fk
import polars as pl

from windmark.core.managers import ClassificationManager
from windmark.core.structs import Hyperparameters


@fk.task
def rebalance_class_labels(
    ledger: str,
    params: Hyperparameters,
) -> ClassificationManager:
    records: dict[str, list[float]] = (
        pl.scan_parquet(ledger)
        .select("target")
        .collect()
        .get_column("target")
        .value_counts()
        .select(labels=pl.col("target"), counts=pl.col("count"))
        .to_dict(as_series=False)
    )

    balancer = ClassificationManager(
        labels=records["labels"],
        counts=records["counts"],
        kappa=params.interpolation_rate,
    )

    balancer.show()

    return balancer
