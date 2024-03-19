import flytekit as fk
import polars as pl
from pytdigest import TDigest
import numpy as np

from source.core.schema import Field


@fk.task
def create_digest_centroids_from_ledger(
    ledger: str,
    field: Field,
    n_slices: int = 10_000
) -> dict[str, np.ndarray]:

    digest = TDigest()

    if field.type != "continuous":
        return {}

    shards = (
        pl.scan_parquet(ledger)
        .select(field.name)
        .filter(pl.col(field.name).is_not_null())
        .collect(streaming=True)
        .iter_slices(n_slices)
    )

    for shard in shards:
        digest.update(shard.get_column(field.name).to_numpy())

    return {field.name: digest.get_centroids()}
