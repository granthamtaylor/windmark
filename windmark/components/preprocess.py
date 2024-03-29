from pathlib import Path

import flytekit as fk
import polars as pl

from windmark.core.managers import SystemManager, SplitManager
from windmark.core.structs import Field


@fk.task
def preprocess_ledger_to_shards(
    ledger: str, manager: SystemManager, slice_size: int
) -> fk.types.directory.FlyteDirectory:
    assert len(manager.schema.fields) > 0

    lf = pl.scan_parquet(ledger)

    def format(field: Field) -> pl.Expr:
        match field.type:
            case "continuous":
                return pl.col(field.name)

            case "temporal":
                return pl.col(field.name).dt.epoch(time_unit="s")

            case "discrete" | "entity":
                return pl.col(field.name).fill_null("[UNK]")

    split = SplitManager(0.5, 0.25, 0.25)

    def assign(column: str) -> pl.Expr:
        hashed = pl.col(column).hash().mul(1 / 305175781)
        seed = hashed.ceil().sub(hashed)

        return (
            pl.when(seed.is_between(*split.ranges["train"]))
            .then(pl.lit("train"))
            .when(seed.is_between(*split.ranges["validate"]))
            .then(pl.lit("validate"))
            .when(seed.is_between(*split.ranges["test"]))
            .then(pl.lit("test"))
            .otherwise(pl.lit("train"))
        )

    outpath = Path(fk.current_context().working_directory) / "lifestreams"
    outpath.mkdir(exist_ok=True)

    label_map: dict[str, int] = {label: index for index, label in enumerate(manager.task.balancer.labels)}

    lifestreams = (
        lf.select(
            *[format(field) for field in manager.schema.fields],
            target=pl.col(manager.schema.target_id).replace(label_map).cast(pl.Int32).fill_null(-1),
            event_id=manager.schema.event_id,
            sequence_id=manager.schema.sequence_id,
            order_by=manager.schema.order_by,
            split=assign(manager.schema.sequence_id),
        )
        .sort("sequence_id", "order_by")
        .group_by("sequence_id", maintain_order=True)
        .agg(
            *[field.name for field in manager.schema.fields],
            size=pl.count().cast(pl.Int32),
            event_ids=pl.col("event_id"),
            target=pl.col("target"),
            split=pl.col("split").last(),
        )
        .collect()
        .iter_slices(slice_size)
    )

    for index, sequence in enumerate(lifestreams):
        sequence.write_avro(outpath / f"{index}.avro", name="lifestream")

    print(f"finished preprocessing {index+1} lifestream files")

    return fk.types.directory.FlyteDirectory(outpath)
