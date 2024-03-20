import flytekit as fk
import polars as pl
from pytdigest import TDigest
import numpy as np

from source.core.schema import Field


@fk.task
def create_digest_centroids_from_ledger(
    ledger: str,
    field: Field,
    slice_size: int = 10_000
) -> dict[str, np.ndarray]:

    digest = TDigest()

    if field.type not in ["continuous", "temporal"]:
        return {}


    def format(field: Field) -> pl.Expr:
        if field.type == 'continuous':
            return pl.col(field.name)
        else:
            return pl.col(field.name).dt.epoch(time_unit="s")

    shards = (
        pl.scan_parquet(ledger)
        .select(format(field))
        .filter(pl.col(field.name).is_not_null())
        .collect(streaming=True)
        .iter_slices(slice_size)
    )

    for shard in shards:
        digest.update(shard.get_column(field.name).to_numpy())

    return {field.name: digest.get_centroids()}
