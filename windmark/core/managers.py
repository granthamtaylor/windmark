import math

from rich.console import Console
from rich.table import Table

from windmark.core.structs import Hyperparameters


class ClassificationManager:
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
        table = Table(title=f"ClassificationManager(kappa={self.kappa:.2%})")

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


class SplitManager:
    def __init__(
        self,
        train: float,
        validate: float,
        test: float,
    ):
        self.train: float = train
        self.validate: float = validate
        self.test: float = test

        for split in [train, validate, test]:
            assert isinstance(split, float)
            assert 0.05 < split < 1.0

        self.ranges: dict[str, tuple[float, float]] = dict(
            train=(0.0, train),
            validate=(train, train + validate),
            test=(train + validate, 1.0),
        )

        assert math.isclose(sum([train, validate, test]), 1.0)


class SequenceManager:
    def __init__(
        self,
        n_sequences: int,
        n_events: int,
        shard_size: int,
        params: Hyperparameters,
        balancer: ClassificationManager,
        split: SplitManager,
    ):
        self.n_sequences: int = n_sequences
        self.n_events: int = n_events
        self.shard_size: int = shard_size

        self.params: Hyperparameters = params
        self.balancer: ClassificationManager = balancer
        self.split: SplitManager = split

        # expected pretraining steps per epoch
        pretraining_steps = int(split.train * n_events * params.pretrain_sample_rate / params.batch_size)

        n_labeled_events = 0
        for label_count, label_sample_rate in zip(balancer.counts, balancer.thresholds):
            n_labeled_events += label_count * label_sample_rate

        # expected finetuning steps per epoch
        finetuning_steps = int(split.train * n_labeled_events * params.finetune_sample_rate / params.batch_size)

        inference_steps = int(split.test * n_events / params.batch_size)

        print(f"Expected number of pretraining steps: {pretraining_steps}")
        print(f"Expected number of finetuning steps: {finetuning_steps}")
        print(f"Expected number of inference steps: {inference_steps}")
