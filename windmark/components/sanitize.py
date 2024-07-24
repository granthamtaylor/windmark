import os

from pathlib import Path

from flytekit.types import directory
from windmark.core.orchestration import task


@task
def sanitize_lifestreams_path(datapath: str) -> directory.FlyteDirectory:
    """
    Sanitizes the lifestreams path by checking if it exists and counting the number of lifestream files.

    Args:
        datapath (str): The path to the lifestreams directory.

    Returns:
        directory.FlyteDirectory: The sanitized lifestreams directory.

    Raises:
        AssertionError: If the lifestreams path does not exist or if no lifestream files are found.
    """

    if not Path(datapath).exists():
        raise FileExistsError("lifestreams path does not exist")

    files = [file for file in os.listdir(datapath) if file.endswith(".ndjson")]

    if len(files) == 0:
        raise FileExistsError("lifestreams path does not have any lifestream files")

    print(f"found {len(files)} lifestream files in '{datapath}'")

    return directory.FlyteDirectory(datapath)
