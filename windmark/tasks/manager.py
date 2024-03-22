import flytekit as fk
import polars as pl

from windmark.core.managers import ClassificationManager, SequenceManager, SplitManager
from windmark.core.structs import Hyperparameters


@fk.task
def create_sequence_manager(
    ledger: str,
    shard_size: int,
    balancer: ClassificationManager,
    params: Hyperparameters,
):
    lf = pl.scan_parquet(ledger)

    n_events = lf.select(pl.len()).collect().item()
    n_sequences = lf.unique(subset=["sequence_id"]).select(pl.len()).collect().item()

    print(n_sequences)
    print(n_events)

    split = SplitManager(0.5, 0.25, 0.25)

    SequenceManager(
        n_sequences=n_sequences,
        n_events=n_events,
        shard_size=shard_size,
        params=params,
        balancer=balancer,
        split=split,
    )
