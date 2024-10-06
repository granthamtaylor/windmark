from windmark.core.orchestration import task
from windmark.core.managers import (
    SchemaManager,
    SupervisedTaskManager,
    SampleManager,
    SplitManager,
    CentroidManager,
    LevelManager,
    SystemManager,
)


@task
def create_system_manager(
    schema: SchemaManager,
    task: SupervisedTaskManager,
    sample: SampleManager,
    split: SplitManager,
    centroids: CentroidManager,
    levelsets: LevelManager,
) -> SystemManager:
    """
    Creates a system state manager to contain multiple child state managers.

    Args:
        schema (SchemaManager): The schema manager object.
        task (SupervisedTaskManager): The supervised task manager object.
        sample (SampleManager): The sample manager object.
        split (SplitManager): The split manager object.
        centroids (CentroidManager): The centroid manager object.
        levelsets (LevelManager): The level manager object.

    Returns:
        SystemManager: The created system state manager.
    """
    manager = SystemManager(
        schema=schema,
        task=task,
        sample=sample,
        split=split,
        centroids=centroids,
        levelsets=levelsets,
    )

    manager.show()

    return manager
