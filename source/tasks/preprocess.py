from pathlib import Path

import flytekit as fk
import polars as pl

from source.core.schema import SPECIAL_TOKENS, Field
from source.core.utils import LabelBalancer


@fk.task
def preprocess_ledger_to_shards(
    ledger: str,
    fields: list[Field],
    shard_size: int,
    balancer: LabelBalancer,
) -> fk.types.directory.FlyteDirectory:

    assert len(fields) > 0
    assert shard_size > 0
    
    lf = pl.scan_parquet(ledger)

    def discretize(column: str) -> pl.Expr:

        return (
            pl.col(column)
            .cast(pl.String)
            .cast(pl.Categorical)
            .to_physical()
            .cast(pl.Int32)
            .add(len(SPECIAL_TOKENS))
            .fill_null(getattr(SPECIAL_TOKENS, "UNK_"))
            .alias(column)
        )

    def format(field: Field) -> pl.Expr:
        match field.type:
            case "continuous":
                return pl.col(field.name)

            case "discrete":
                return discretize(field.name)

            case "entity":
                return pl.col(field.name)

    outpath = Path(fk.current_context().working_directory) / "lifestreams"
    outpath.mkdir(exist_ok=True)
    
    label_map: dict[str, int] = {label: index for index, label in enumerate(balancer.labels)}
    
    lifestreams = (
        lf.select(
            *[format(field) for field in fields],
            pl.col("target").replace(label_map).cast(pl.Int32).fill_null(-1),
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
        .collect()
        .iter_slices(shard_size)
    )

    for index, shard in enumerate(lifestreams):
        shard.write_avro(outpath / f"shard-{index}.avro", name="lifestream")
    
    print(f'finished preprocessing {index+1} lifestream files')

    return fk.types.directory.FlyteDirectory(outpath)
