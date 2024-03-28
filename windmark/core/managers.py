import math

from rich.console import Console
from rich.table import Table
from pytdigest import TDigest


from windmark.core.structs import Hyperparameters, Field, Centroid


class SchemaManager:
    def __init__(
        self,
        sequence_id: str,
        event_id: str,
        order_by: str,
        target_id: str,
        **fields: str,
    ):
        assert len(fields) > 1, "must pass in at least two fields"

        assert isinstance(sequence_id, str)
        assert isinstance(event_id, str)
        assert isinstance(order_by, str)
        assert isinstance(target_id, str)

        self.sequence_id: str = sequence_id
        self.event_id: str = event_id
        self.order_by: str = order_by
        self.target_id: str = target_id

        self.fields: list[Field] = []

        for field in fields.items():
            self.fields.append(Field(*field))

    def __len__(self) -> int:
        return len(self.fields)


class BalanceManager:
    labels: list[str]
    counts: list[int]
    total: int
    values: list[float]
    interpolation: list[float]
    thresholds: list[float]
    weights: list[float]
    kappa: float

    def __init__(self, labels: list[str], counts: list[int], kappa: float):
        for count in counts:
            assert count > 0

        assert len(labels) == len(counts)

        size: int = len(labels)
        null: float = 1 / size

        self.labels = labels
        self.counts = counts
        self.total = sum(counts)
        self.values = [count / self.total for count in counts]

        assert math.isclose(sum(self.values), 1.0)

        assert 0.0 <= kappa <= 1.0

        self.interpolation = [kappa * null + (1 - kappa) * value for value in self.values]

        assert math.isclose(sum(self.interpolation), 1.0)

        ratio = [value / interpol for interpol, value in zip(self.values, self.interpolation)]

        self.thresholds: list[float] = list(map(lambda x: x / max(ratio), ratio))

        weights = list(map(lambda x: sum(ratio) / x, self.interpolation))
        self.weights = list(map(lambda x: x / min(weights), weights))

        self.kappa = kappa

    def show(self):
        table = Table(title=f"SupervisedTaskManager(kappa={self.kappa:.2%})")

        table.add_column("Metric", justify="right", style="cyan", no_wrap=True)

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
        table.add_row("Observation Distribution", *format_percent(self.interpolation))
        table.add_row("Marginal Sample Rate", *format_percent(self.thresholds))
        table.add_row("Loss Weights", *format_numbers(self.weights))

        console = Console()
        console.print(table)


class SupervisedTaskManager:
    def __init__(
        self,
        task: str,
        n_targets: int,
        balancer: BalanceManager,
    ):
        assert task in ["classification", "regression"]
        assert balancer is not None
        assert n_targets > 1

        self.n_targets: int = n_targets
        self.balancer: BalanceManager = balancer


class SplitManager:
    def __init__(
        self,
        train: float,
        validate: float,
        test: float,
    ):
        self.splits: dict[str, float] = {}

        self.splits["train"] = train
        self.splits["validate"] = validate
        self.splits["test"] = test

        for split in [train, validate, test]:
            assert isinstance(split, float)
            assert 0.05 < split < 1.0

        self.ranges: dict[str, tuple[float, float]] = dict(
            train=(0.0, train),
            validate=(train, train + validate),
            test=(train + validate, 1.0),
        )

        assert math.isclose(sum([train, validate, test]), 1.0)


class SampleManager:
    def __init__(
        self,
        n_events: int,
        params: Hyperparameters,
        task: SupervisedTaskManager,
        split: SplitManager,
    ):
        balancer = task.balancer

        # expected finetuning steps per epoch
        self.pretraining: dict[str, float] = {}
        for subset in ["train", "validate", "test"]:
            sample_rate = (params.n_steps * params.batch_size) / (split.splits[subset] * n_events)
            assert (
                sample_rate < 1.0
            ), f"not enough observations to create {params.n_steps} batches for split {subset} during pretraining"
            self.pretraining[subset] = sample_rate

        n_targets = 0
        for label_count, label_sample_rate in zip(balancer.counts, balancer.thresholds):
            n_targets += label_count * label_sample_rate

        # expected finetuning steps per epoch
        self.finetuning: dict[str, float] = {}
        for subset in ["train", "validate", "test"]:
            sample_rate = (params.n_steps * params.batch_size) / (split.splits[subset] * n_targets)
            assert (
                sample_rate < 1.0
            ), f"not enough observations to create {params.n_steps} batches for split {subset} during finetuning"
            self.finetuning[subset] = sample_rate

        self.inference: int = int(split.splits["test"] * n_events / params.batch_size)

        print(self.pretraining)
        print(self.finetuning)


class CentroidManager:
    def __init__(self, centroids: list[Centroid]):
        for centroid in centroids:
            assert isinstance(centroid, Centroid)

        self.centroids: list[Centroid] = [centroid for centroid in centroids if centroid.is_valid]

    @property
    def digests(self) -> dict[str, TDigest]:
        digests: dict[str, TDigest] = {}

        for centroid in self.centroids:
            digests[centroid.name] = TDigest.of_centroids(centroid.array)

        return digests


class SequenceManager:
    def __init__(
        self,
        schema: SchemaManager,
        task: SupervisedTaskManager,
        sample: SampleManager,
        split: SplitManager,
        centroids: CentroidManager,
    ):
        self.schema: SchemaManager = schema
        self.task: SupervisedTaskManager = task
        self.sample: SampleManager = sample
        self.split: SplitManager = split
        self.centroids: CentroidManager = centroids

    @property
    def digests(self):
        return self.centroids.digests
