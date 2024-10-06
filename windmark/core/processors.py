from pathlib import Path
from typing import Any
import os
from collections import Counter

from beartype import beartype
from beartype.typing import Callable
from mpire import WorkerPool
import msgspec
import numpy as np
from pytdigest import TDigest


@beartype
def digest(resources: dict[str, Any], worker_id: int) -> list[list[float]]:
    """
    Process the given resources and calculate the centroids using TDigest.

    Args:
        resources (dict[str, Any]): A dictionary containing the necessary resources.
        worker_id (int): The ID of the worker processing the resources.

    Returns:
        list[list[float]]: A list of centroids calculated using TDigest.
    """
    tdigest = TDigest()

    decoder = msgspec.json.Decoder(dict[str, Any])

    for index, filename in enumerate(resources["filenames"]):
        if index % resources["n_workers"] == worker_id:
            with open(resources["path"] / filename, "rb") as file:
                for line in file:
                    inputs = decoder.decode(line)[resources["key"]]
                    tdigest.update(np.array(inputs, dtype=float))

    return tdigest.get_centroids().tolist()


@beartype
def count(resources: dict[str, Any], worker_id: int) -> Counter:
    """
    Counts the occurrences of items in the input resources.

    Args:
        resources (dict[str, Any]): A dictionary containing the necessary resources.
        worker_id (int): The ID of the worker processing the resources.

    Returns:
        Counter: A Counter object containing the counts of the items.

    """
    counter = Counter()

    decoder = msgspec.json.Decoder(dict[str, Any])

    for index, filename in enumerate(resources["filenames"]):
        if index % resources["n_workers"] == worker_id:
            with open(resources["path"] / filename, "rb") as file:
                for line in file:
                    inputs = decoder.decode(line)[resources["key"]]
                    if isinstance(inputs, list):
                        counter.update(inputs)
                    else:
                        counter.update([inputs])

    return counter


@beartype
def multithread(process: Callable, key: str, path: Path) -> list[Any]:
    """
    Execute a given process function in parallel using multiple threads.

    Args:
        process (Callable): The function to be executed in parallel.
        key (str): A key parameter for the process function.
        path (Path): The path to the directory containing the files to be processed.

    Returns:
        list[Any]: A list of results returned by the process function for each worker thread.
    """

    filenames = [filename for filename in os.listdir(path) if filename.endswith(".ndjson")]

    n_workers = os.cpu_count()

    resources: dict[str, Any] = dict(n_workers=n_workers, key=key, path=path, filenames=filenames)

    with WorkerPool(n_jobs=n_workers, shared_objects=resources) as pool:
        results = pool.map(process, range(n_workers))

        return results
