import flytekit as fk
import polars as pl

from windmark.core.managers import SchemaManager, SupervisedTaskManager, BalanceManager
from windmark.core.constructs import Hyperparameters


@fk.task
def create_task_manager(
    ledger: str,
    schema: SchemaManager,
    params: Hyperparameters,
) -> SupervisedTaskManager:
    lf = pl.scan_parquet(ledger)

    records: dict[str, list] = (
        lf.select(schema.target_id)
        .collect()
        .get_column(schema.target_id)
        .value_counts()
        .select(labels=pl.col(schema.target_id), counts=pl.col("count"))
        .to_dict(as_series=False)
    )

    assert len(records["labels"]) <= 100

    labels: list[str] = records["labels"]
    counts: list[int] = records["counts"]

    balancer = BalanceManager(labels=labels, counts=counts, kappa=params.interpolation_rate)

    return SupervisedTaskManager(task="classification", n_targets=len(records["labels"]), balancer=balancer)
