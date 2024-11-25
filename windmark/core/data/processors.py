# Copyright Grantham Taylor.

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
def compare(resources: dict[str, Any], worker_id: int):
    """
    Compares the lengths of the inputs and indices in the given resources.

    Args:
        resources (dict[str, Any]): A dictionary containing the necessary resources.
        worker_id (int): The ID of the worker processing the resources.
    """

    decoder = msgspec.json.Decoder(dict[str, Any])

    schema = resources["schema"]
    key = resources["key"]
    is_static = resources["is_static"]

    for index, filename in enumerate(resources["filenames"]):
        if index % resources["n_workers"] == worker_id:
            with open(resources["path"] / filename, "rb") as file:
                for line in file:
                    sequence = decoder.decode(line)
                    inputs: list[Any] = sequence[key]
                    indices: list[Any] = sequence[schema.event_id]

                    if is_static:
                        if isinstance(inputs, list):
                            raise ValueError(f"Inputs must be a scalar, got '{type(inputs)}' instead")

                        continue

                    if not isinstance(inputs, list):
                        raise ValueError(f"Inputs must be a list, got '{type(inputs)}' instead")

                    if not isinstance(indices, list):
                        raise ValueError(f"Event ID must be a list, got '{type(indices)}' instead")

                    if len(inputs) != len(indices):
                        sequence_id = sequence[schema.sequence_id]
                        raise ValueError(
                            "The lengths of the inputs and indices must match"
                            f"sequence '{sequence_id}' in '{filename}' do not match"
                            f"inputs: {len(inputs)}, indices: {len(indices)}"
                        )


@beartype
def multithread(process: Callable, path: Path, **kwargs) -> list[Any]:
    """
    Execute a given process function in parallel using multiple threads.

    Args:
        process (Callable): The function to be executed in parallel.
        path (Path): The path to the directory containing the files to be processed.
        kwargs (Any): Additional keyword arguments to be passed to the process function.

    Returns:
        list[Any]: A list of results returned by the process function for each worker thread.
    """

    filenames = [filename for filename in os.listdir(path) if filename.endswith(".ndjson")]

    n_workers = os.cpu_count()

    resources: dict[str, Any] = dict(filenames=filenames, n_workers=n_workers, path=path, **kwargs)

    with WorkerPool(n_jobs=n_workers, shared_objects=resources) as pool:
        results = pool.map(process, range(n_workers))

        return results
