from windmark.core.orchestration import task
from windmark.core.constructs.general import Centroid, LevelSet
from windmark.core.managers import (
    SchemaManager,
    SupervisedTaskManager,
    SampleManager,
    SplitManager,
    SystemManager,
    CentroidManager,
    LevelManager,
)


@task
def create_system_manager(
    schema: SchemaManager,
    task: SupervisedTaskManager,
    sample: SampleManager,
    split: SplitManager,
    centroids: list[Centroid | None],
    levelsets: list[LevelSet | None],
) -> SystemManager:
    """
    Creates a system state manager to contain multiple child state managers.

    Args:
        schema (SchemaManager): The schema manager object.
        task (SupervisedTaskManager): The supervised task manager object.
        sample (SampleManager): The sample manager object.
        split (SplitManager): The split manager object.
        centroids (list[Centroid|None]): The centroid objects.
        levelsets (list[LevelSet|None]): The levelset objects.

    Returns:
        SystemManager: The created system state manager.
    """
    manager = SystemManager(
        schema=schema,
        task=task,
        sample=sample,
        split=split,
        centroids=CentroidManager(centroids),
        levelsets=LevelManager(levelsets),
    )

    manager.show()

    return manager
