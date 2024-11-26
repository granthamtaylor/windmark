# Copyright Grantham Taylor.

import math
import dataclasses
import functools

import numpy as np
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from pytdigest import TDigest

from windmark.core.constructs.general import Centroid, LevelSet, FieldRequest


@dataclasses.dataclass
class SchemaManager:
    """Manages the schema for Windmark data."""

    sequence_id: str
    event_id: str
    split_id: str
    target_id: str
    fields: list[FieldRequest]

    # TODO I could probably clean this up
    @classmethod
    def new(
        cls,
        sequence_id: str,
        event_id: str,
        split_id: str,
        target_id: str,
        fields: dict[str, str],
    ) -> "SchemaManager":
        """
        Create a new instance of SchemaManager.

        Args:
            cls: The class object.
            sequence_id: The sequence ID.
            event_id: The event ID.
            split_id: The split ID.
            target_id: The target ID.
            fields: The fields of the schema.

        Returns:
            An instance of SchemaManager.

        Raises:
            ValueError: If no dynamic fields are included.

        """
        fields: list[FieldRequest] = [FieldRequest.new(name, field) for name, field in fields.items()]

        return cls(
            sequence_id=sequence_id,
            event_id=event_id,
            split_id=split_id,
            target_id=target_id,
            fields=fields,
        )

    def __post_init__(self):
        """Perform additional initialization after the object has been created."""

        if len([field for field in self.fields if not field.type.is_static]) < 1:
            raise ValueError("include at least one dynamic field")

        assert len(self.fields) > 1, "must pass in at least two fields"

        reserved_names: set[str] = {self.sequence_id, self.event_id, self.split_id, self.target_id}

        assert len(reserved_names) == 4

        for field in self.fields:
            assert field.name not in reserved_names, f"field name {field.name} is invalid (already reserved)"

    def __len__(self) -> int:
        """
        Returns the number of fields in the manager.

        Returns:
            int: The number of fields in the manager.
        """
        return len(self.fields)

    @functools.cached_property
    def static(self) -> list[FieldRequest]:
        """
        Returns a list of FieldRequest objects that are marked as static.

        Returns:
            list[FieldRequest]: A list of FieldRequest objects that are marked as static.
        """
        return [field for field in self.fields if field.is_static]

    @functools.cached_property
    def dynamic(self) -> list[FieldRequest]:
        """
        Returns a list of dynamic FieldRequest objects.

        This method filters the list of fields and returns only the ones that are not static.

        Returns:
            list[FieldRequest]: A list of dynamic FieldRequest objects.
        """
        return [field for field in self.fields if not field.is_static]


@dataclasses.dataclass
class BalanceManager:
    """A class that manages the balance of class labels in a dataset."""

    labels: list[str]
    counts: list[int]
    interpolation_rate: float
    unlabeled: int

    total: int = dataclasses.field(init=False)
    values: list[float] = dataclasses.field(init=False)
    interpolation: list[float] = dataclasses.field(init=False)
    thresholds: list[float] = dataclasses.field(init=False)
    weights: list[float] = dataclasses.field(init=False)

    def __post_init__(self):
        """
        Initializes the class and calculates the necessary attributes.
        """
        for count in self.counts:
            assert count > 0

        assert len(self.labels) == len(self.counts)

        size: int = len(self.labels)
        null: float = 1 / size

        self.total = sum(self.counts)
        self.values = [count / self.total for count in self.counts]

        assert math.isclose(sum(self.values), 1.0)

        assert 0.0 <= self.interpolation_rate <= 1.0

        self.interpolation = [
            self.interpolation_rate * null + (1 - self.interpolation_rate) * value for value in self.values
        ]

        assert math.isclose(sum(self.interpolation), 1.0)

        ratio = [value / interpol for interpol, value in zip(self.values, self.interpolation)]

        self.thresholds: list[float] = list(map(lambda x: x / max(ratio), ratio))

        weights = list(map(lambda x: sum(ratio) / x, self.interpolation))
        self.weights = list(map(lambda x: x / min(weights), weights))

    def show(self):
        """
        Displays the balance manager information in a table.
        """

        console = Console()

        table = Table(title=f"Balance Manager (interpolation rate = {self.interpolation_rate:.2%})")

        table.add_column("Class Labels", justify="right", style="cyan", no_wrap=True)
        table.add_column("Label Counts", style="cyan", no_wrap=True)
        table.add_column("Population Distribution", style="cyan", no_wrap=True)
        table.add_column("Sample Rate", style="cyan", no_wrap=True)
        table.add_column("Modified Distribution", style="cyan", no_wrap=True)
        table.add_column("Loss Weight", style="cyan", no_wrap=True)

        for index in range(len(self.labels)):
            table.add_row(
                self.labels[index],
                f"{self.counts[index]:,}",
                f"{self.values[index]:.4%}",
                f"{self.interpolation[index]:.4%}",
                f"{self.thresholds[index]:.4%}",
                f"{self.weights[index]:.4}",
            )

        console.print(table)

    @functools.cached_property
    def mapping(self) -> dict[str, int]:
        """
        Returns a mapping of class labels to their indices.
        """
        return {label: index for index, label in enumerate(self.labels)}

    @functools.cached_property
    def sample_rates_mapping(self) -> dict[str, float]:
        """
        Returns the sample rates for each class label.
        """
        return {label: rate for label, rate in zip(self.labels, self.thresholds)}

    @functools.cached_property
    def label_mapping(self) -> dict[str, int]:
        """
        Returns a mapping of class labels to their indices.
        """
        return {label: index for index, label in enumerate(self.labels)}

    @property
    def n_events(self) -> int:
        """
        Returns the total number of events, including unlabeled instances.
        """
        return self.total + self.unlabeled


@dataclasses.dataclass
class SupervisedTaskManager:
    """
    A manager class for supervised learning tasks.

    Raises:
        AssertionError: If the task is not "classification" or "regression".
        AssertionError: If the number of targets is less than or equal to 1.
    """

    task: str
    n_targets: int
    balancer: BalanceManager

    def __post_init__(self):
        assert self.task in ["classification", "regression"]
        assert self.n_targets > 1


@dataclasses.dataclass
class SplitManager:
    """A class that manages the splits for a dataset."""

    train: int
    validate: int
    test: int

    def __post_init__(self):
        """
        Perform additional initialization after the object is created.

        This method is automatically called by the `dataclasses` module after the object is created.
        It can be used to perform any additional initialization steps that are required.
        """

        for split in [self.train, self.validate, self.test]:
            assert isinstance(split, int)

    def __getitem__(self, split: str) -> float:
        """
        Get the proportion of a specific split.

        Args:
            split (str): The name of the split ("train", "validate", or "test").

        Returns:
            float: The proportion of the specified split.
        """
        assert split in ["train", "validate", "test"]

        total: int = self.total

        splits = dict(train=self.train, validate=self.validate, test=self.test)

        return splits[split] / total

    @functools.cached_property
    def total(self) -> int:
        """
        Calculate the total size of all splits.

        Returns:
            int: The total size of all splits.
        """
        return sum([self.train, self.validate, self.test])


@dataclasses.dataclass
class SampleManager:
    """A class that manages the sampling rates for pretraining and finetuning in a machine learning task."""

    batch_size: int
    n_pretrain_steps: int
    n_finetune_steps: int
    task: SupervisedTaskManager
    split: SplitManager

    def __post_init__(self):
        """
        Initializes the SampleManager and calculates the sample rates for pretraining and finetuning.

        Raises:
            AssertionError: If the sample rate exceeds 1.0 for any subset during pretraining or finetuning.
        """
        n_events = self.task.balancer.n_events

        def warn(mode: str, n_steps: int, split: str):
            return f"There not enough observations to create {n_steps} batches for split '{split}' during {mode}."

        # expected finetuning steps per epoch
        self.pretraining: dict[str, float] = {}
        for subset in ["train", "validate", "test"]:
            sample_rate = (self.n_pretrain_steps * self.batch_size) / (self.split[subset] * n_events)
            assert sample_rate < 1.0, warn(mode="pretraining", n_steps=self.n_pretrain_steps, split=subset)
            self.pretraining[subset] = sample_rate

        n_targets = 0
        for label_count, label_sample_rate in zip(self.task.balancer.counts, self.task.balancer.thresholds):
            n_targets += label_count * label_sample_rate

        # expected finetuning steps per epoch
        self.finetuning: dict[str, float] = {}
        for subset in ["train", "validate", "test"]:
            sample_rate = (self.n_finetune_steps * self.batch_size) / (self.split[subset] * n_targets)
            assert sample_rate < 1.0, warn(mode="finetuning", n_steps=self.n_finetune_steps, split=subset)
            self.finetuning[subset] = sample_rate

    def show(self) -> None:
        """
        Displays the sample rates for pretraining and finetuning in a table format.
        """

        console = Console()

        def render(mode: str) -> Table:
            assert mode in ["finetune", "pretrain"]

            if mode == "pretrain":
                title = "Pretraining Sample Manager"
                rates = self.pretraining
            else:
                title = "Finetuning Sample Manager"
                rates = self.finetuning

            table = Table(title=title)

            table.add_column("Strata", justify="right", style="cyan", no_wrap=True)
            table.add_column("Sample Rate", justify="right", style="cyan", no_wrap=True)

            for strata, sample_rate in rates.items():
                table.add_row(strata, f"{sample_rate:.4%}")

            return table

        n_events = self.split.total

        n_inference_batches: int = int(self.split["test"] * n_events / self.batch_size)

        renderables = Group(
            render("pretrain"),
            render("finetune"),
            f"inference batches: {n_inference_batches:,}",
        )

        console.print(Panel.fit(renderables, title="Sample Managers"))


@dataclasses.dataclass
class CentroidManager:
    """A class that manages centroids and provides methods to calculate digests and display information."""

    centroids: list[Centroid | None]

    def __post_init__(self):
        self.centroids = [centroid for centroid in self.centroids if centroid is not None]

    @functools.cached_property
    def digests(self) -> dict[str, TDigest]:
        """
        Returns a dictionary of TDigest objects.

        Returns:
            dict[str, TDigest]: A dictionary where the keys are centroid names and the values are TDigest objects.
        """
        digests: dict[str, TDigest] = {}

        for centroid in self.centroids:
            digests[centroid.name] = TDigest.of_centroids(np.array(centroid.array))

        return digests

    def show(self):
        """
        Display the centroid manager's digest information in a table format.

        This method prints a table that shows the field names and their corresponding
        values at different percentiles. The percentiles are defined by the `percentiles`
        list.

        Args:
            None

        Returns:
            None
        """

        console = Console()

        percentiles = [0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0]

        table = Table(title="Centroid Manager")

        table.add_column("Field Names", justify="right", style="cyan", no_wrap=True)

        for percentile in percentiles:
            table.add_column(f"Q({percentile:.2})", style="cyan", no_wrap=True)

        for fieldname, digest in self.digests.items():
            values = []

            for percentile in percentiles:
                value = digest.inverse_cdf(percentile)  # type: ignore

                values.append(f"{value:.3f}")

            table.add_row(fieldname, *values)

        console.print(table)

    def __repr__(self) -> str:
        return f"CentroidManager({[centroid.name for centroid in self.centroids]})"


@dataclasses.dataclass
class LevelManager:
    """
    A class that manages levels and mappings for different fields.
    """

    levelsets: list[LevelSet | None]
    mappings: dict[str, dict[str, int]] = dataclasses.field(init=False)

    def __post_init__(self):
        levelsets = [levelset for levelset in self.levelsets if levelset is not None]
        self.mappings = {levelset.name: levelset.mapping for levelset in levelsets}

    def __getitem__(self, field: FieldRequest) -> dict[str, int]:
        """
        Get the mapping for a specific field.

        Args:
            field (FieldRequest): The field for which to get the mapping.

        Returns:
            dict[str, int]: The mapping for the specified field.
        """
        assert isinstance(field, FieldRequest)
        return self.mappings[field.name]

    def show(self):
        """
        Display the level manager information in a table format.
        """

        console = Console()

        table = Table(title="Level Manager")

        table.add_column("Field Names", justify="right", style="cyan", no_wrap=True)
        table.add_column("Depth", style="cyan", no_wrap=True)
        table.add_column("Levels", style="cyan", no_wrap=False)

        for field, mapping in self.mappings.items():
            size: int = len(mapping) - 1

            levels: list[str] = [level for level in list(mapping.keys())]

            max_levels = 50

            formatted_levels = (", ").join(levels[: min(max_levels, size)])

            if size >= max_levels:
                formatted_levels += "..."

            table.add_row(field, f"{size:,}", formatted_levels)

        console.print(table)

    def __repr__(self) -> str:
        return f"LevelManager({[name for name in self.mappings.keys()]})"


@dataclasses.dataclass
class SystemManager:
    """The SystemManager class manages various components of the system."""

    schema: SchemaManager
    task: SupervisedTaskManager
    sample: SampleManager
    split: SplitManager
    centroids: CentroidManager
    levelsets: LevelManager

    def show(self):
        """
        Displays information for each presenter in the system.
        """
        presenters = [
            self.task.balancer,
            self.sample,
            self.centroids,
            self.levelsets,
        ]

        for presenter in presenters:
            presenter.show()
