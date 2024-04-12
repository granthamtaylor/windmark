import flytekit as fk
import polars as pl

from windmark.core.managers import SupervisedTaskManager, SplitManager, SampleManager
from windmark.core.constructs import Hyperparameters


@fk.task
def create_sample_manager(
    ledger: str,
    params: Hyperparameters,
    task: SupervisedTaskManager,
    split: SplitManager,
) -> SampleManager:
    n_events = pl.scan_parquet(ledger).select(pl.len()).collect().item()

    return SampleManager(n_events=n_events, params=params, task=task, split=split)
