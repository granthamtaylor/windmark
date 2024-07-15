from pathlib import Path
from typing import Any, Callable
import os
from collections import Counter

from mpire import WorkerPool
import fastavro
import numpy as np
from pytdigest import TDigest


def digest(resources: dict[str, Any], worker_id: int) -> list[TDigest]:
    digest = TDigest()

    for index, filename in enumerate(resources["filenames"]):
        if index % resources["n_workers"] == worker_id:
            with open(resources["path"] / filename, "rb") as f:
                reader = fastavro.reader(f)
                for sequence in reader:
                    inputs = sequence[resources["key"]]
                    digest.update(np.array(inputs, dtype=float))

    return digest.get_centroids().tolist()


def count(resources: dict[str, Any], worker_id: int) -> Counter:
    counter = Counter()

    for index, filename in enumerate(resources["filenames"]):
        if index % resources["n_workers"] == worker_id:
            with open(resources["path"] / filename, "rb") as f:
                reader = fastavro.reader(f)
                for sequence in reader:
                    inputs = sequence[resources["key"]]
                    if isinstance(inputs, list):
                        counter.update(inputs)
                    else:
                        counter.update([inputs])

    return counter


def multithread(n_workers: int, process: Callable, key: str, path: Path) -> list[TDigest | Counter]:
    filenames = [filename for filename in os.listdir(path) if filename.endswith(".avro")]

    resources: dict[str, Any] = dict(
        n_workers=n_workers,
        key=key,
        path=path,
        filenames=filenames,
    )

    with WorkerPool(n_jobs=n_workers, shared_objects=resources) as pool:
        results = pool.map(process, range(n_workers))

    return results
