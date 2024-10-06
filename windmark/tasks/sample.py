from windmark.core.managers import SupervisedTaskManager, SplitManager, SampleManager
from windmark.core.orchestration import task


@task
def create_sample_manager(
    task: SupervisedTaskManager,
    batch_size: int,
    n_pretrain_steps: int,
    n_finetune_steps: int,
    split: SplitManager,
) -> SampleManager:
    """
    Creates a SampleManager object to determine effective observation sampling rates.

    Args:
        task (SupervisedTaskManager): The task manager for the supervised task.
        batch_size (int): The batch size for training.
        n_pretrain_steps (int): The number of pre-training steps.
        n_finetune_steps (int): The number of fine-tuning steps.
        split (SplitManager): The split manager for the dataset.

    Returns:
        SampleManager: The created SampleManager object.
    """

    return SampleManager(
        batch_size=batch_size,
        n_pretrain_steps=n_pretrain_steps,
        n_finetune_steps=n_finetune_steps,
        task=task,
        split=split,
    )
