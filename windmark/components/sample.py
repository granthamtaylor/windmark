import polars as pl

from windmark.core.managers import SupervisedTaskManager, SplitManager, SampleManager
from windmark.core.orchestration import task


@task
def create_sample_manager(
    ledger: str,
    task: SupervisedTaskManager,
    batch_size: int,
    n_pretrain_steps: int,
    n_finetune_steps: int,
    split: SplitManager,
) -> SampleManager:
    n_events = pl.scan_parquet(ledger).select(pl.len()).collect().item()

    return SampleManager(
        n_events=n_events,
        batch_size=batch_size,
        n_pretrain_steps=n_pretrain_steps,
        n_finetune_steps=n_finetune_steps,
        task=task,
        split=split,
    )
