import polars as pl
from pytdigest import TDigest

from windmark.core.constructs.general import Centroid, FieldRequest
from windmark.core.orchestration import task


@task
def create_digest_centroids_from_ledger(ledger: str, field: FieldRequest, slice_size: int = 10_000) -> Centroid:
    digest = TDigest()

    if field.type not in ["continuous", "temporal"]:
        return Centroid.empty(field.name)

    def format(field: FieldRequest) -> pl.Expr:
        if field.type == "continuous":
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

    return Centroid.from_digest(field.name, digest=digest)
