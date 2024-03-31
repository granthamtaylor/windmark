import flytekit as fk


from windmark.core.managers import (
    SchemaManager,
    SupervisedTaskManager,
    SampleManager,
    SplitManager,
    CentroidManager,
    LevelManager,
    SystemManager,
)


@fk.task
def create_system_manager(
    schema: SchemaManager,
    task: SupervisedTaskManager,
    sample: SampleManager,
    split: SplitManager,
    centroids: CentroidManager,
    levelsets: LevelManager,
) -> SystemManager:
    return SystemManager(schema=schema, task=task, sample=sample, split=split, centroids=centroids, levelsets=levelsets)
