from pathlib import Path
from functools import reduce

from pytdigest import TDigest
import numpy as np
from flytekit.types import directory

from windmark.core.constructs.general import Centroid, FieldRequest, FieldType
from windmark.core.orchestration import task
from windmark.core.processors import multithread, digest


@task
def create_digest_centroids_from_lifestream(
    lifestreams: directory.FlyteDirectory,
    field: FieldRequest,
    n_workers: int,
) -> Centroid:
    if field.type not in [FieldType.Number, FieldType.Numbers, FieldType.Quantiles, FieldType.Quantile]:
        return Centroid.empty(field.name)

    print(f'- creating state manager for field "{field.name}"')

    path = Path(lifestreams.path)

    results = multithread(n_workers=n_workers, process=digest, key=field.name, path=path)

    digests = [TDigest.of_centroids(np.array(digest)) for digest in results]

    output = reduce(lambda a, b: a + b, digests)

    output.force_merge()

    return Centroid.from_digest(field.name, digest=output)
