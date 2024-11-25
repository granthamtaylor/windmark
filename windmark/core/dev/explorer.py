# Copyright Grantham Taylor.

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

from windmark.core.constructs.managers import SchemaManager


class LifestreamSampler:
    """
    A class representing a sampler for a lifestream dataset.

    This class provides methods to sample and retrieve dynamic and static data from a lifestream dataset.
    """

    def __init__(self) -> None:
        config = Path(os.getcwd()) / "config"
        cwd = Path(os.path.realpath(__file__)).parent

        with initialize(version_base=None, config_path=os.path.relpath(config, cwd)):
            manifest = compose(config_name="config")["data"]

        datapath = Path(manifest["path"])

        files = os.listdir(datapath)

        shards = [file for file in files if file.endswith("ndjson")]

        self.schema = SchemaManager.new(**manifest["structure"], **manifest["fields"])
        self.df = pl.read_ndjson(datapath / random.choice(shards)).sample(n=1)

    @property
    def dynamic(self) -> tuple[list[str], list[tuple[str, ...]]]:
        """
        Retrieve the dynamic data from the lifestream dataset.

        Returns:
            tuple[list[str], list[tuple[str, ...]]]: A tuple containing the column names and rows of the dynamic data.
        """
        dynamic = (
            self.df.select(self.schema.event_id, self.schema.target_id, *[field.name for field in self.schema.dynamic])
            .explode(pl.all())
            .select(pl.all().cast(pl.String))
        )

        return dynamic.columns, dynamic.rows()

    @property
    def static(self) -> tuple[list[str], list[tuple[str, ...]]]:
        """
        Retrieve the static data from the lifestream dataset.

        Returns:
            tuple[list[str], list[tuple[str, ...]]]: A tuple containing the column names and rows of the static data.
        """
        static = self.df.select(
            pl.col(self.schema.event_id).list.len().alias("_n_events"),
            cs.string(),
            cs.numeric(),
            cs.temporal(),
            cs.boolean(),
        ).unpivot(variable_name="name")

        return static.columns, static.rows()


class TableApp(App):
    """
    Represents an application for displaying tables.

    This class provides functionality for composing and displaying tables with static and dynamic fields.
    It also includes an action to resample the data and update the tables accordingly.
    """

    BINDINGS = [
        Binding(key="q", action="quit", description="Quit"),
        Binding(key="r", action="resample", description="Resample"),
    ]

    def compose(self) -> ComposeResult:
        """
        Composes the UI elements for the application.

        Returns:
            A ComposeResult object representing the composed UI elements.
        """
        with TabbedContent():
            with TabPane("Static Fields"):
                with HorizontalScroll():
                    yield DataTable(id="static")

            with TabPane("Dynamic Fields"):
                with HorizontalScroll():
                    yield DataTable(id="dynamic")

        yield Footer()

    def action_resample(self) -> None:
        """
        Resamples the data and updates the tables.

        This method retrieves the resampled data from a LifestreamSampler object and updates the
        corresponding DataTable objects with the new data.
        """
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
        """
        Event handler called when the application is mounted.

        This method is responsible for initializing the application and calling the action_resample method
        to populate the tables with initial data.
        """
        self.action_resample()


app = TableApp()
if __name__ == "__main__":
    app.run()
