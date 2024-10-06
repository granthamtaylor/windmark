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
) -> Centroid:
    """
    Creates digest centroids from the given lifestreams.

    Args:
        lifestreams (directory.FlyteDirectory): The directory containing the lifestreams.
        field (FieldRequest): The field to create digest centroids for.

    Returns:
        Centroid: The digest created for numeric field.
    """

    if field.type not in [FieldType.Number, FieldType.Numbers, FieldType.Quantiles, FieldType.Quantile]:
        return Centroid.empty(field.name)

    print(f'- creating state manager for field "{field.name}"')

    path = Path(lifestreams.path)

    results = multithread(process=digest, key=field.name, path=path)

    digests = [TDigest.of_centroids(np.array(digest)) for digest in results]

    output = reduce(lambda a, b: a + b, digests)

    output.force_merge()

    return Centroid.from_digest(field.name, digest=output)
