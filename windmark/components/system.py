from datetime import datetime
import string
import random

import flytekit as fk
from faker import Faker

from windmark.core.managers import (
    SchemaManager,
    SupervisedTaskManager,
    SampleManager,
    SplitManager,
    CentroidManager,
    LevelManager,
    SystemManager,
)


def label() -> str:
    fake = Faker()

    # sorting
    date = datetime.now().strftime("%Y-%m-%d")

    # human readability
    address = fake.street_name().replace(" ", "-").lower()

    # quick searching
    hashtag = ("").join(random.choice(string.ascii_uppercase) for _ in range(4))

    return f"{date}:{address}:{hashtag}"


@fk.task
def create_system_manager(
    schema: SchemaManager,
    task: SupervisedTaskManager,
    sample: SampleManager,
    split: SplitManager,
    centroids: CentroidManager,
    levelsets: LevelManager,
) -> SystemManager:
    version = label()

    manager = SystemManager(
        version=version,
        schema=schema,
        task=task,
        sample=sample,
        split=split,
        centroids=centroids,
        levelsets=levelsets,
    )

    manager.show()

    return manager
