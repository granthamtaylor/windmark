import math
import datetime
import dataclasses
import functools

import numpy as np
from rich.console import Console, Group
from rich.table import Table
from rich import print
from rich.panel import Panel
from pytdigest import TDigest
from mashumaro.mixins.json import DataClassJSONMixin

from windmark.core.constructs import Hyperparameters, Field, Centroid, LevelSet

console = Console()


@dataclasses.dataclass
class ArtifactManager(DataClassJSONMixin):
    ledger: str
    model: str
    predictions: str


@dataclasses.dataclass
class SchemaManager(DataClassJSONMixin):
    sequence_id: str
    event_id: str
    order_by: str
    target_id: str
    fields: list[Field]

    @classmethod
    def create(
        cls,
        sequence_id: str,
        event_id: str,
        order_by: str,
        target_id: str,
        **fieldinfo: str,
    ) -> "SchemaManager":
        fields: list[Field] = []

        for field in fieldinfo.items():
            fields.append(Field(*field))

        return cls(
            sequence_id=sequence_id,
            event_id=event_id,
            order_by=order_by,
            target_id=target_id,
            fields=fields,
        )

    def __post_init__(self):
        assert len(self.fields) > 1, "must pass in at least two fields"

        reserved_names: set[str] = {
            self.sequence_id,
            self.event_id,
            self.order_by,
            self.target_id,
        }

        assert len(reserved_names) == 4

        for field in self.fields:
            assert field.name not in reserved_names, f"field name {field.name} is invalid (already reserved)"

    def __len__(self) -> int:
        return len(self.fields)


@dataclasses.dataclass
class BalanceManager(DataClassJSONMixin):
    labels: list[str]
    counts: list[int]
    kappa: float

    total: int = dataclasses.field(init=False)
    values: list[float] = dataclasses.field(init=False)
    interpolation: list[float] = dataclasses.field(init=False)
    thresholds: list[float] = dataclasses.field(init=False)
    weights: list[float] = dataclasses.field(init=False)

    def __post_init__(self):
        for count in self.counts:
            assert count > 0

        assert len(self.labels) == len(self.counts)

        size: int = len(self.labels)
        null: float = 1 / size

        self.total = sum(self.counts)
        self.values = [count / self.total for count in self.counts]

        assert math.isclose(sum(self.values), 1.0)

        assert 0.0 <= self.kappa <= 1.0

        self.interpolation = [self.kappa * null + (1 - self.kappa) * value for value in self.values]

        assert math.isclose(sum(self.interpolation), 1.0)

        ratio = [value / interpol for interpol, value in zip(self.values, self.interpolation)]

        self.thresholds: list[float] = list(map(lambda x: x / max(ratio), ratio))

        weights = list(map(lambda x: sum(ratio) / x, self.interpolation))
        self.weights = list(map(lambda x: x / min(weights), weights))

        self.kappa = self.kappa

    def show(self):
        table = Table(title=f"Balance Manager (kappa={self.kappa:.2%})")

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


@dataclasses.dataclass
class SupervisedTaskManager(DataClassJSONMixin):
    task: str
    n_targets: int
    balancer: BalanceManager

    def __post_init__(self):
        assert self.task in ["classification", "regression"]
        assert self.n_targets > 1


@dataclasses.dataclass
class SplitManager(DataClassJSONMixin):
    train: float
    validate: float
    test: float

    def __post_init__(self):
        assert math.isclose(sum([self.train, self.validate, self.test]), 1.0)

        for split in [self.train, self.validate, self.test]:
            assert isinstance(split, float)
            assert 0.05 < split < 1.0

    def __getitem__(self, split: str) -> float:
        assert split in ["train", "validate", "test"]

        splits = dict(
            train=self.train,
            validate=self.validate,
            test=self.test,
        )

        return splits[split]

    @property
    def ranges(self) -> dict[str, tuple[float, float]]:
        return dict(
            train=(0.0, self.train),
            validate=(self.train, self.train + self.validate),
            test=(self.train + self.validate, 1.0),
        )


@dataclasses.dataclass
class SampleManager(DataClassJSONMixin):
    n_events: int
    params: Hyperparameters
    task: SupervisedTaskManager
    split: SplitManager

    def __post_init__(self):
        def warn(mode: str, n_steps: int, split: str):
            return f"There not enough observations to create {n_steps} batches for split '{split}' during {mode}."

        # expected finetuning steps per epoch
        self.pretraining: dict[str, float] = {}
        for subset in ["train", "validate", "test"]:
            sample_rate = (self.params.n_pretrain_steps * self.params.batch_size) / (self.split[subset] * self.n_events)
            assert sample_rate < 1.0, warn(mode="pretraining", n_steps=self.params.n_pretrain_steps, split=subset)
            self.pretraining[subset] = sample_rate

        n_targets = 0
        for label_count, label_sample_rate in zip(self.task.balancer.counts, self.task.balancer.thresholds):
            n_targets += label_count * label_sample_rate

        # expected finetuning steps per epoch
        self.finetuning: dict[str, float] = {}
        for subset in ["train", "validate", "test"]:
            sample_rate = (self.params.n_finetune_steps * self.params.batch_size) / (self.split[subset] * n_targets)
            assert sample_rate < 1.0, warn(mode="finetuning", n_steps=self.params.n_finetune_steps, split=subset)
            self.finetuning[subset] = sample_rate

    def show(self) -> None:
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

        n_inference_batches: int = int(self.split["test"] * self.n_events / self.params.batch_size)

        renderables = Group(
            render("pretrain"),
            render("finetune"),
            f"inference batches: {n_inference_batches:,}",
        )

        console.print(Panel.fit(renderables, title="Sample Managers"))


@dataclasses.dataclass
class CentroidManager(DataClassJSONMixin):
    centroids: list[Centroid]

    def __post_init__(self):
        self.centroids = [centroid for centroid in self.centroids if centroid.is_valid]

    @functools.cached_property
    def digests(self) -> dict[str, TDigest]:
        digests: dict[str, TDigest] = {}

        for centroid in self.centroids:
            digests[centroid.name] = TDigest.of_centroids(np.array(centroid.array))

        return digests

    def show(self, schema: SchemaManager):
        percentiles = [0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0]

        types: dict[str, str] = {field.name: field.type for field in schema.fields}

        table = Table(title="Centroid Manager")

        table.add_column("Field Names", justify="right", style="cyan", no_wrap=True)

        for percentile in percentiles:
            table.add_column(f"Q({percentile:.2})", style="cyan", no_wrap=True)

        for field, digest in self.digests.items():
            values = []

            for percentile in percentiles:
                value = digest.inverse_cdf(percentile)  # type: ignore

                if types[field] == "continuous":
                    values.append(f"{value:.3f}")
                elif types[field] == "temporal":
                    values.append(datetime.datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M"))  # type: ignore
                else:
                    raise NotImplementedError

            table.add_row(field, *values)

        console.print(table)


@dataclasses.dataclass
class LevelManager(DataClassJSONMixin):
    levelsets: list[LevelSet]
    mappings: dict[str, dict[str, int]] = dataclasses.field(init=False)

    def __post_init__(self):
        self.mappings = {levelset.name: levelset.mapping for levelset in self.levelsets if levelset.is_valid}

    def get_size(self, field: Field) -> int:
        assert isinstance(field, Field)

        # ! need to subtract one because "[UNK]" is hardcoded into enum
        return len(self.mappings[field.name]) - 1

    def __getitem__(self, field: Field) -> dict[str, int]:
        assert isinstance(field, Field)
        return self.mappings[field.name]

    def show(self):
        table = Table(title="Level Manager")

        table.add_column("Field Names", justify="right", style="cyan", no_wrap=True)

        table.add_column("Depth", style="cyan", no_wrap=True)
        table.add_column("Levels", style="cyan", no_wrap=False)

        for field, mapping in self.mappings.items():
            size: int = len(mapping) - 1

            levels: list[str] = [level for level in list(mapping.keys()) if level != "[UNK]"]

            max_levels = 50

            formatted_levels = (", ").join(levels[: min(max_levels, size)])

            if size >= max_levels:
                formatted_levels += "..."

            table.add_row(field, f"{size:,}", formatted_levels)

        console.print(table)


@dataclasses.dataclass
class SystemManager(DataClassJSONMixin):
    version: str
    schema: SchemaManager
    task: SupervisedTaskManager
    sample: SampleManager
    split: SplitManager
    centroids: CentroidManager
    levelsets: LevelManager

    def show(self):
        print(Panel.fit(f"[cyan]{self.version}", title="Training Model", padding=(1, 3)))

        self.task.balancer.show()
        self.sample.show()
        self.centroids.show(schema=self.schema)
        self.levelsets.show()
