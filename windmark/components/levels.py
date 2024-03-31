import flytekit as fk
import polars as pl

from windmark.core.managers import Field, LevelSet


@fk.task
def create_unique_levels_from_ledger(ledger: str, field: Field) -> LevelSet:
    if field.type not in ["discrete"]:
        return LevelSet.empty(name=field.name)

    levels: list[str] = (
        pl.scan_parquet(ledger)
        .select(pl.col(field.name).cast(pl.String))
        .filter(pl.col(field.name).is_not_null())
        .unique()
        .collect(streaming=True)
        .get_column(field.name)
        .to_list()
    )

    return LevelSet.from_levels(name=field.name, levels=levels)
