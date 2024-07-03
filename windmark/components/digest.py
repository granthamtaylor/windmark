import os

from pytdigest import TDigest
import numpy as np
from flytekit.types import directory
import fastavro

from windmark.core.constructs.general import Centroid, FieldRequest, FieldType
from windmark.core.orchestration import task


@task
def create_digest_centroids_from_lifestream(lifestreams: directory.FlyteDirectory, field: FieldRequest) -> Centroid:
    if field.type not in [FieldType.Number, FieldType.Numbers]:
        return Centroid.empty(field.name)

    print(f"starting to create digests for field {field.name}")

    digest = TDigest()

    for filename in os.listdir(lifestreams.path):
        if filename.endswith(".avro"):
            with open(f"{lifestreams.path}/{filename}", "rb") as f:
                reader = fastavro.reader(f)
                for sequence in reader:
                    inputs = sequence[field.name]
                    if inputs is not None:
                        digest.update(np.array(inputs))

    return Centroid.from_digest(field.name, digest=digest)
