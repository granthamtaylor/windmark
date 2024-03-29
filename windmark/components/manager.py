import flytekit as fk


from windmark.core.managers import (
    SchemaManager,
    SupervisedTaskManager,
    SampleManager,
    SplitManager,
    CentroidManager,
    SystemManager,
)


@fk.task
def create_sequence_manager(
    schema: SchemaManager,
    task: SupervisedTaskManager,
    sample: SampleManager,
    split: SplitManager,
    centroids: CentroidManager,
) -> SystemManager:
    return SystemManager(schema=schema, task=task, sample=sample, split=split, centroids=centroids)
