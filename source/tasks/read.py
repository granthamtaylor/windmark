import os

from flytekit import task
import polars as pl


@task
def read_ledger_from_parquet(datapath: str | os.PathLike) -> pl.DataFrame:
    assert os.path.exists(datapath)

    df = pl.read_parquet(datapath)

    return df
