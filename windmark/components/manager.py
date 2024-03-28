import flytekit as fk


from windmark.core.managers import (
    SchemaManager,
    SupervisedTaskManager,
    SampleManager,
    SplitManager,
    CentroidManager,
    SequenceManager,
)


@fk.task
def create_sequence_manager(
    schema: SchemaManager,
    task: SupervisedTaskManager,
    sample: SampleManager,
    split: SplitManager,
    centroids: CentroidManager,
) -> SequenceManager:
    return SequenceManager(schema=schema, task=task, sample=sample, split=split, centroids=centroids)
