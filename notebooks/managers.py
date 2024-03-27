from typing import TypeAlias
import math

import numpy as np
from pytdigest import TDigest


class Field:
    pass


class Hyperparameters:
    pass


class Schema:
    def __init__(
        self,
        sequence_id: str,
        event_id: str,
        order_by: str,
        target_id: str,
        **fields: str,
    ):
        assert len(fields) > 1, "must pass in at least two fields"

        self.fields: list[Field] = []

        for name, dtype in fields.items():
            self.fields.append(Field(name=name, dtype=dtype))

    def __len__(self) -> int:
        return len(self.fields)


class BalanceManager:
    pass


class ClassificationTaskManager:
    n_targets: int
    balancer: BalanceManager


class RegressionTaskManager:
    n_targets: int = 1


SupervisedTaskManager: TypeAlias = ClassificationTaskManager | RegressionTaskManager


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
        balancer: BalanceManager,
        split: SplitManager,
    ):
        # expected finetuning steps per epoch
        self.pretraining = {}
        for split in ["train", "validation", "test"]:
            sample_rate = (params.n_steps * params.batch_size) / (balancer.splits[split] * n_events)
            assert (
                sample_rate < 1.0
            ), f"not enough observations to create {params.n_steps} batches for split {split} during pretraining"
            self.pretraining[split] = sample_rate

        n_targets = 0
        for label_count, label_sample_rate in zip(balancer.counts, balancer.thresholds):
            n_targets += label_count * label_sample_rate

        # expected finetuning steps per epoch
        self.finetuning = {}
        for split in ["train", "validation", "test"]:
            sample_rate = (params.n_steps * params.batch_size) / (balancer.splits[split] * n_targets)
            assert (
                sample_rate < 1.0
            ), f"not enough observations to create {params.n_steps} batches for split {split} during finetuning"
            self.finetuning[split] = sample_rate

        self.inference = int(split.test * n_events / params.batch_size)


class Centroid:
    def __init__(self, name: str, digest: TDigest):
        assert isinstance(name, str)
        assert isinstance(digest, TDigest)

        self.name: str = name
        self.centroid: np.ndarray = digest.get_centroids()


class CentroidManager:
    def __init__(self, centroids: list[Centroid]):
        for centroid in centroids:
            assert isinstance(centroid, Centroid)

        self.centroids = centroids

    @property
    def digests(self) -> dict[str, TDigest]:
        digests: dict[str, TDigest] = {}

        for centroid in self.centroid:
            digests[centroid.name] = TDigest.of_centroids(centroid)

        return digests


class SequenceManagers:
    schema: Schema
    task: SupervisedTaskManager
    sample: SampleManager
    split: SplitManager
    centroids: CentroidManager

    @property
    def digests(self):
        return self.centroids.digests
