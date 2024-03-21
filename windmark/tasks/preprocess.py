from pathlib import Path

import flytekit as fk
import polars as pl

from windmark.core.schema import SPECIAL_TOKENS, Field
from windmark.core.utils import LabelBalancer, SplitManager


@fk.task
def preprocess_ledger_to_shards(
    ledger: str,
    fields: list[Field],
    balancer: LabelBalancer,
) -> fk.types.directory.FlyteDirectory:

    assert len(fields) > 0
    
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

            case "temporal":
                return pl.col(field.name).dt.epoch(time_unit="s")

            case "discrete":
                return discretize(field.name)

            case "entity":
                return pl.col(field.name).fill_null('UNK_')

    split = SplitManager(0.5, 0.25, 0.25)

    def assign(column: str) -> pl.Expr:

        hashed = pl.col(column).hash().mul(1/305175781)
        seed = hashed.ceil().sub(hashed)
        
        return (
            pl
            .when(seed.is_between(*split.ranges['train'])).then(pl.lit('train'))
            .when(seed.is_between(*split.ranges['validate'])).then(pl.lit('validate'))
            .when(seed.is_between(*split.ranges['test'])).then(pl.lit('test'))
            .otherwise(pl.lit('train'))
        )

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
            split=assign('sequence_id')
        )
        .sort("sequence_id", "event_order")
        .group_by("sequence_id", maintain_order=True)
        .agg(
            *[field.name for field in fields],
            size=pl.count().cast(pl.Int32),
            event_ids=pl.col("event_id"),
            target=pl.col("target"),
            split=pl.col("split").last()
        )
        .collect()
        .iter_slices(1)
    )

    for index, sequence in enumerate(lifestreams):
        split = sequence.get_column('split').item()
        sequence.write_avro(outpath / f"{split}-{index}.avro", name="lifestream")
    
    print(f'finished preprocessing {index+1} lifestream files')

    return fk.types.directory.FlyteDirectory(outpath)
