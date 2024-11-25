# Copyright Grantham Taylor.

from windmark.core.constructs.managers import SupervisedTaskManager, SplitManager, SampleManager
from windmark.core.constructs.general import Hyperparameters
from windmark.orchestration.environments import context


@context.default
def create_sample_manager(
    task: SupervisedTaskManager,
    params: Hyperparameters,
    split: SplitManager,
) -> SampleManager:
    """
    Creates a SampleManager object to determine effective observation sampling rates.

    Args:
        task (SupervisedTaskManager): The task manager for the supervised task.
        params (Hyperparameters): model hyperparameters.
        split (SplitManager): The split manager for the dataset.

    Returns:
        SampleManager: The created SampleManager object.
    """

    return SampleManager(
        batch_size=params.batch_size,
        n_pretrain_steps=params.n_pretrain_steps,
        n_finetune_steps=params.n_finetune_steps,
        task=task,
        split=split,
    )
