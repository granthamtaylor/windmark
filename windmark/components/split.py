import os
from collections import Counter

from flytekit.types import directory
import fastavro

from windmark.core.managers import SchemaManager, SplitManager
from windmark.core.orchestration import task


@task
def create_split_manager(
    lifestreams: directory.FlyteDirectory,
    schema: SchemaManager,
) -> SplitManager:
    counter = Counter()

    for filename in os.listdir(lifestreams.path):
        if filename.endswith(".avro"):
            with open(f"{lifestreams.path}/{filename}", "rb") as f:
                reader = fastavro.reader(f)
                for sequence in reader:
                    counter.update([sequence[schema.split_id]])

    splits: dict[str, int] = dict(counter)

    for split in list(splits.keys()):
        assert split in ["train", "test", "validate"]

    proportions: dict[str, int] = dict(splits)

    return SplitManager(**proportions)
