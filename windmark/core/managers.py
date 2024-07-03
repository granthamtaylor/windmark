import math
import dataclasses
import functools
from datetime import datetime
import string
import random

from faker import Faker
import numpy as np
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from pytdigest import TDigest
from mashumaro.mixins.json import DataClassJSONMixin

from windmark.core.constructs.general import Centroid, LevelSet, FieldRequest, FieldType

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
    split_id: str
    target_id: str
    fields: list[FieldRequest]

    @classmethod
    def new(
        cls,
        sequence_id: str,
        event_id: str,
        split_id: str,
        target_id: str,
        **fieldinfo: FieldType | str,
    ) -> "SchemaManager":
        fields: list[FieldRequest] = []

        for field in fieldinfo.items():
            fields.append(FieldRequest.new(*field))

        assert len([field for field in fields if not field.type.is_static]) > 0, "include at least one dynamic field"

        return cls(
            sequence_id=sequence_id,
            event_id=event_id,
            split_id=split_id,
            target_id=target_id,
            fields=fields,
        )

    def __post_init__(self):
        assert len(self.fields) > 1, "must pass in at least two fields"

        reserved_names: set[str] = {
            self.sequence_id,
            self.event_id,
            self.split_id,
            self.target_id,
        }

        assert len(reserved_names) == 4

        for field in self.fields:
            assert field.name not in reserved_names, f"field name {field.name} is invalid (already reserved)"

    def __len__(self) -> int:
        return len(self.fields)

    @functools.cached_property
    def static(self) -> list[FieldRequest]:
        return [field for field in self.fields if field.is_static]

    @functools.cached_property
    def dynamic(self) -> list[FieldRequest]:
        return [field for field in self.fields if not field.is_static]


@dataclasses.dataclass
class BalanceManager(DataClassJSONMixin):
    labels: list[str]
    counts: list[int]
    kappa: float
    unlabeled: int

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

    @functools.cached_property
    def mapping(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.labels)}

    @property
    def n_events(self) -> int:
        return self.total + self.unlabeled


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
    train: int
    validate: int
    test: int

    def __post_init__(self):
        for split in [self.train, self.validate, self.test]:
            assert isinstance(split, int)
            assert split / self.total > 0.05

    def __getitem__(self, split: str) -> float:
        assert split in ["train", "validate", "test"]

        total: int = self.total

        splits = dict(train=self.train, validate=self.validate, test=self.test)

        return splits[split] / total

    @functools.cached_property
    def total(self) -> int:
        return sum([self.train, self.validate, self.test])


@dataclasses.dataclass
class SampleManager(DataClassJSONMixin):
    batch_size: int
    n_pretrain_steps: int
    n_finetune_steps: int
    task: SupervisedTaskManager
    split: SplitManager

    def __post_init__(self):
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

    def show(self):
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


@dataclasses.dataclass
class LevelManager(DataClassJSONMixin):
    levelsets: list[LevelSet]
    mappings: dict[str, dict[str, int]] = dataclasses.field(init=False)

    def __post_init__(self):
        self.mappings = {levelset.name: levelset.mapping for levelset in self.levelsets if levelset.is_valid}

    def get_size(self, field: FieldRequest) -> int:
        assert isinstance(field, FieldRequest)
        return len(self.mappings[field.name])

    def __getitem__(self, field: FieldRequest) -> dict[str, int]:
        assert isinstance(field, FieldRequest)
        return self.mappings[field.name]

    def show(self):
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


@dataclasses.dataclass
class SystemManager(DataClassJSONMixin):
    schema: SchemaManager
    task: SupervisedTaskManager
    sample: SampleManager
    split: SplitManager
    centroids: CentroidManager
    levelsets: LevelManager

    def show(self):
        presenters = [
            self.task.balancer,
            self.sample,
            self.centroids,
            self.levelsets,
        ]

        for presenter in presenters:
            presenter.show()


class LabelManager:
    @classmethod
    def version(cls) -> str:
        fake = Faker()

        address = fake.street_name().replace(" ", "-").lower()

        hashtag = ("").join(random.choice(string.ascii_uppercase) for _ in range(4))

        return f"{address}:{hashtag}"

    @classmethod
    def from_path(cls, pathname: str, add_date: bool = True) -> str:
        filename = pathname.split("/")[-1]
        version = filename.split(".")[0]

        if add_date:
            date = datetime.now().strftime("%Y-%m-%d %H:%M")
            return f"{date}:{version}"
        else:
            return version
