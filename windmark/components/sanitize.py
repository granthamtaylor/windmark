import os

from pathlib import Path

from flytekit.types import directory
from windmark.core.orchestration import task


@task
def sanitize_lifestreams_path(datapath: str) -> directory.FlyteDirectory:
    assert Path(datapath).exists(), "lifestreams path does not exist"

    files = [file for file in os.listdir(datapath) if file.endswith(".ndjson")]

    assert len(files) > 0

    print(f"found {len(files)} lifestream files in '{datapath}'")

    return directory.FlyteDirectory(datapath)
