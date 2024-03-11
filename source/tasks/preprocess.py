from pathlib import Path

import flytekit as fl
import polars as pl

from source.core.schema import SPECIAL_TOKENS, Field


@fl.task
def preprocess_ledger_to_shards(
    ledger: pl.DataFrame,
    fields: list[Field],
    n_shards: int,
) -> fl.types.directory.FlyteDirectory:
    assert len(fields) > 0
    assert n_shards > 0

    def discretize(column: str) -> pl.Expr:
        return pl.col(column).cast(pl.String).cast(pl.Categorical).to_physical().cast(pl.Int32).alias(column)

    def format(field: Field) -> pl.Expr:
        match field.dtype:
            case "continuous":
                return pl.col(field.name)

            case "discrete":
                return discretize(field.name).add(len(SPECIAL_TOKENS)).fill_null(getattr(SPECIAL_TOKENS, "UNK_"))

            case "entity":
                return pl.col(field.name)

    shard_size = ledger.get_column("sequence_id").unique().count() // n_shards

    outpath = Path(fl.current_context().working_directory) / "lifestreams"
    outpath.mkdir(exist_ok=True)
    print(outpath)

    lifestreams = (
        ledger.select(
            *[format(field) for field in fields],
            discretize("target"),
            "event_id",
            "sequence_id",
            "event_order",
        )
        .sort("sequence_id", "event_order")
        .group_by("sequence_id", maintain_order=True)
        .agg(
            *[field.name for field in fields],
            size=pl.count().cast(pl.Int32),
            event_ids=pl.col("event_id"),
            target=pl.col("target"),
        )
        .iter_slices(shard_size)
    )

    for index, shard in enumerate(lifestreams):
        shard.write_avro(outpath / f"shard-{index}.avro", name="lifestream")

    return fl.types.directory.FlyteDirectory(outpath)
