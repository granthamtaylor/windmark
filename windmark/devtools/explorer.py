import random
import os
from pathlib import Path

from hydra import compose, initialize
import polars as pl
import polars.selectors as cs
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import HorizontalScroll
from textual.widgets import DataTable, TabbedContent, TabPane, Footer

from windmark.core.managers import SchemaManager


class LifestreamSampler:
    def __init__(self) -> None:
        with initialize(version_base=None, config_path="../../config"):
            manifest = compose(config_name="config")["data"]

        datapath = Path(manifest["path"])

        files = os.listdir(datapath)

        shards = [file for file in files if file.endswith("ndjson")]

        self.schema = SchemaManager.new(**manifest["structure"], **manifest["fields"])
        self.df = pl.read_ndjson(datapath / random.choice(shards)).sample(n=1)

    @property
    def dynamic(self) -> tuple[list[str], list[tuple[str, ...]]]:
        dynamic = (
            self.df.select(self.schema.event_id, self.schema.target_id, *[field.name for field in self.schema.dynamic])
            .explode(pl.all())
            .select(pl.all().cast(pl.String))
        )

        return dynamic.columns, dynamic.rows()

    @property
    def static(self) -> tuple[list[str], list[tuple[str, ...]]]:
        static = self.df.select(
            pl.col(self.schema.event_id).list.len().alias("_n_events"),
            cs.string(),
            cs.numeric(),
            cs.temporal(),
            cs.boolean(),
        ).unpivot(variable_name="name")

        return static.columns, static.rows()


class TableApp(App):
    BINDINGS = [
        Binding(key="q", action="quit", description="Quit"),
        Binding(key="r", action="reload", description="Reload"),
    ]

    def compose(self) -> ComposeResult:
        with TabbedContent():
            with TabPane("Dynamic Fields"):
                with HorizontalScroll():
                    yield DataTable(id="dynamic")
            with TabPane("Static Fields"):
                with HorizontalScroll():
                    yield DataTable(id="static")

        yield Footer()

    def action_reload(self) -> None:
        sample = LifestreamSampler()

        columns, records = sample.dynamic
        table = self.query_one("#dynamic")
        table.clear(columns=True)
        table.add_columns(*columns)
        table.add_rows(records)

        columns, records = sample.static
        table = self.query_one("#static")
        table.clear(columns=True)
        table.add_columns(*columns)
        table.add_rows(records)

    def on_mount(self) -> None:
        self.action_reload()


app = TableApp()
if __name__ == "__main__":
    app.run()
