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
