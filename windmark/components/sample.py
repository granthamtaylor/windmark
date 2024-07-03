from flytekit.types import directory

from windmark.core.managers import SupervisedTaskManager, SplitManager, SampleManager, SchemaManager
from windmark.core.orchestration import task


@task
def create_sample_manager(
    lifestreams: directory.FlyteDirectory,
    schema: SchemaManager,
    task: SupervisedTaskManager,
    batch_size: int,
    n_pretrain_steps: int,
    n_finetune_steps: int,
    split: SplitManager,
) -> SampleManager:
    return SampleManager(
        batch_size=batch_size,
        n_pretrain_steps=n_pretrain_steps,
        n_finetune_steps=n_finetune_steps,
        task=task,
        split=split,
    )
