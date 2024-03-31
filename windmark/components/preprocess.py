from pathlib import Path
from zlib import crc32

import flytekit as fk
import polars as pl
from rich.console import Console

from windmark.core.managers import SystemManager
from windmark.core.structs import Field


console = Console()


@fk.task
def preprocess_ledger_to_shards(
    ledger: str, manager: SystemManager, slice_size: int
) -> fk.types.directory.FlyteDirectory:
    assert len(manager.schema.fields) > 0

    lf = pl.scan_parquet(ledger)

    def format(field: Field) -> pl.Expr:
        match field.type:
            case "continuous":
                return pl.col(field.name).cast(pl.Float32)

            case "temporal":
                return pl.col(field.name).dt.epoch(time_unit="s").cast(pl.Float32)

            case "discrete" | "entity":
                return pl.col(field.name).fill_null("[UNK]")

            case _:
                raise NotImplementedError

    def assign_split(column: str) -> pl.Expr:
        ranges = manager.split.ranges

        seed = (
            pl.col(column)
            .cast(pl.String)
            .map_elements(lambda x: float(crc32(str.encode(x)) & 0xFFFFFFFF), return_dtype=pl.Float32)
            .mul(1 / 2**32)
        )

        return (
            pl.when(seed.is_between(*ranges["train"]))
            .then(pl.lit("train"))
            .when(seed.is_between(*ranges["validate"]))
            .then(pl.lit("validate"))
            .when(seed.is_between(*ranges["test"]))
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
            split=assign_split(manager.schema.sequence_id),
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

    index = 0

    for sequence in lifestreams:
        sequence.write_avro(outpath / f"{index}.avro", name="lifestream")
        index += 1

    console.print(f"[red]INFO:[/] finished preprocessing [bold]{index+1}[/] lifestream shards")

    return fk.types.directory.FlyteDirectory(outpath)
