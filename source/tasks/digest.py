import flytekit as fk
import polars as pl
from tdigest import TDigest

from source.core.schema import Field


@fk.task
def create_digest_from_ledger(
    ledger: str,
    field: Field,
    n_slices: int = 10_000
) -> dict[str, TDigest]:
    digest = TDigest()

    if field.dtype != "continuous":
        return {}

    shards = (
        pl.scan_parquet(ledger)
        .select(field.name)
        .filter(pl.col(field.name).is_not_null())
        .collect(streaming=True)
        .iter_slices(n_slices)
    )

    for shard in shards:
        digest.batch_update(shard.get_column(field.name).to_numpy().astype(float))

    return {field.name: digest}
