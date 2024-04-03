import math
import dataclasses
import functools

import numpy as np
from rich.console import Console
from rich.table import Table
from pytdigest import TDigest
from mashumaro.mixins.json import DataClassJSONMixin

from windmark.core.structs import Hyperparameters, Field, Centroid, LevelSet

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

        reserved_names = set(
            [
                self.sequence_id,
                self.event_id,
                self.order_by,
                self.target_id,
            ]
        )

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
        table = Table(title=f"BalanceManager(kappa={self.kappa:.2%})")

        table.add_column("Class Labels", justify="right", style="cyan", no_wrap=True)

        for label in self.labels:
            table.add_column(f'"{label}"', style="magenta")

        def format_percent(values: list[float]) -> list[str]:
            return list(map(lambda x: f"{x:.4%}", values))

        def format_numbers(values: list[float]) -> list[str]:
            return list(map(lambda x: f"{x:.4}", values))

        def format_integers(values: list[float]) -> list[str]:
            return list(map(lambda x: f"{x:,}", values))

        table.add_row("Label Counts", *format_integers(self.counts))
        table.add_row("Population Distribution", *format_percent(self.values))
        table.add_row("Modified Distribution", *format_percent(self.interpolation))
        table.add_row("Marginal Sample Rate", *format_percent(self.thresholds))
        table.add_row("Loss Weights", *format_numbers(self.weights))

        console.print(table)


@dataclasses.dataclass
class SupervisedTaskManager(DataClassJSONMixin):
    task: str
    n_targets: int
    balancer: BalanceManager

    def __post_init__(self):
        assert self.task in ["classification", "regression"]
        assert self.n_targets > 1


# TODO is this getting recalculated anywhere that is performance critical?
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
        def message(mode: str, n_steps: int, split: str):
            return (
                f"There not enough observations" f" to create {n_steps} batches for" f" split '{split}' during {mode}."
            )

        # expected finetuning steps per epoch
        self.pretraining: dict[str, float] = {}
        for subset in ["train", "validate", "test"]:
            # n_steps = sample_rate * n_events * split_rate / batch_size
            # n_steps * batch_size = sample_rate * n_events * split_rate
            # (n_steps * batch_size) / (n_events * split_rate) = sample_rate

            sample_rate = (self.params.n_steps * self.params.batch_size) / (self.split[subset] * self.n_events)
            assert sample_rate < 1.0, message(mode="pretraining", n_steps=self.params.n_steps, split=subset)
            self.pretraining[subset] = sample_rate

        n_targets = 0
        for label_count, label_sample_rate in zip(self.task.balancer.counts, self.task.balancer.thresholds):
            n_targets += label_count * label_sample_rate

        # expected finetuning steps per epoch
        self.finetuning: dict[str, float] = {}
        for subset in ["train", "validate", "test"]:
            sample_rate = (self.params.n_steps * self.params.batch_size) / (self.split[subset] * n_targets)
            assert sample_rate < 1.0, message(mode="finetuning", n_steps=self.params.n_steps, split=subset)
            self.finetuning[subset] = sample_rate

    def show(self) -> None:
        labeled_inference: int = int(self.split["test"] * sum(self.task.balancer.counts) / self.params.batch_size)
        total_inference: int = int(self.split["test"] * self.n_events / self.params.batch_size)

        print(self.pretraining)
        print(self.finetuning)
        print(f"{total_inference} observations in test split ({labeled_inference} are labeled)")


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


@dataclasses.dataclass
class LevelManager(DataClassJSONMixin):
    levelsets: list[LevelSet]
    mapping: dict[str, dict[str, int]] = dataclasses.field(init=False)

    def __post_init__(self):
        self.mapping = {levelset.name: levelset.mapping for levelset in self.levelsets if levelset.is_valid}

    def get_size(self, field: Field) -> int:
        assert isinstance(field, Field)
        # ! need to subtract one because "UNK" is hardcoded into enum
        return len(self.mapping[field.name]) - 1

    def __getitem__(self, field: Field) -> dict[str, int]:
        assert isinstance(field, Field)
        return self.mapping[field.name]


@dataclasses.dataclass
class SystemManager(DataClassJSONMixin):
    schema: SchemaManager
    task: SupervisedTaskManager
    sample: SampleManager
    split: SplitManager
    centroids: CentroidManager
    levelsets: LevelManager
