import random
from functools import partial
from typing import TypeAlias, Any

import fastavro
import numpy as np
import torch
from pytdigest import TDigest
from tensordict import TensorDict
from torchdata import datapipes

from windmark.core.managers import SystemManager
from windmark.core.constructs import (
    Field,
    ContinuousField,
    DiscreteField,
    EntityField,
    Hyperparameters,
    SupervisedData,
    PretrainingData,
    SequenceData,
    Tokens,
    TemporalField,
)


def read(filename):
    with open(filename, "rb") as f:
        reader = fastavro.reader(f)
        records = [record for record in reader]

    return records


def subset(sequence: dict, split: str) -> bool:
    return sequence["split"] == split


AnnotationType: TypeAlias = tuple[str, str, int]
FieldType: TypeAlias = dict[str, Any]


def sample(
    sequence: dict,
    params: Hyperparameters,
    manager: SystemManager,
    split: str,
    mode: str,
) -> list[tuple[AnnotationType, FieldType]]:
    observations = []

    assert split == sequence["split"]

    for event in range(sequence["size"]):
        if mode == "pretrain":
            if manager.sample.pretraining[split] < random.random():
                continue

            label = -1

        elif mode == "finetune":
            label: int = sequence["target"][event]

            if (label is None) or (label == -1):
                continue

            # finetuning test data should not be downsampled
            if split != "test":
                if manager.sample.finetuning[split] < random.random():
                    continue

                if manager.task.balancer.thresholds[label] < random.random():
                    continue

        elif mode == "inference":
            label: int = sequence["target"][event]

        else:
            raise ValueError

        window = slice(max(0, event - params.n_context), event)

        annotations: AnnotationType = (
            str(sequence["sequence_id"]),
            str(sequence["event_ids"][event]),
            label,
        )

        fields = {}

        for field in manager.schema.fields:
            fields[field.name] = sequence[field.name][window]

        observations.append((annotations, fields))

    return observations


class ContextProcessor:
    offset: int = len(Tokens)

    def __init__(self, manager: SystemManager, params: Hyperparameters):
        self.manager: SystemManager = manager
        self.params: Hyperparameters = params

    def process(self, observation: tuple[AnnotationType, FieldType]) -> tuple[AnnotationType, FieldType]:
        annotations, fields = observation

        for field in self.manager.schema.fields:
            values = fields[field.name]

            match field.type:
                case "discrete":
                    processed = self.tokenize(values=values, field=field)

                case "entity":
                    processed = self.hash(values=values)

                case "continuous" | "temporal":
                    processed = self.cdf(values=values, field=field)

                case _:
                    raise NotImplementedError

            fields[field.name] = processed

        return annotations, fields

    def tokenize(self, values: list[str], field: Field) -> list[int]:
        mapping = self.manager.levelsets[field]

        return list(map(lambda value: mapping[value], values))

    def hash(self, values: list[str]) -> list[int]:
        unique: set[str] = set(values)

        integers = random.sample(range(self.offset, self.params.n_context + self.offset), len(unique))

        mapping = dict(zip(unique, integers))

        mapping.update({"[UNK]": Tokens.UNK})

        return list(map(lambda value: mapping[value], values))

    def cdf(self, values: list[float], field: Field) -> np.ndarray:
        digest: TDigest = self.manager.centroids.digests[field.name]
        array = np.array(values, dtype=np.float64)
        return digest.cdf(array)


def tensorfield(
    observation: tuple[AnnotationType, FieldType],
    params: Hyperparameters,
    manager: SystemManager,
) -> tuple[AnnotationType, TensorDict]:
    annotations, fields = observation

    output = {}

    tensorclasses = dict(
        discrete=DiscreteField,
        continuous=ContinuousField,
        entity=EntityField,
        temporal=TemporalField,
    )

    for field in manager.schema.fields:
        values = fields[field.name]
        tensorclass = tensorclasses[field.type]
        output[field.name] = tensorclass.new(values, params=params)

    return annotations, TensorDict(output, batch_size=1)


def package(
    observation: tuple[AnnotationType, TensorDict],
    params: Hyperparameters,
    manager: SystemManager,
    mode: str,
) -> SequenceData:
    (*meta, label), fields = observation

    assert mode in ("pretrain", "finetune", "inference")

    if mode in ["finetune", "inference"]:
        return SupervisedData.new(inputs=fields, targets=torch.tensor(label), meta=tuple(meta))

    N, L = (1, params.n_context)

    targets = {}

    is_event_masked = torch.rand(N, L).lt(params.p_mask_event)

    for field in manager.schema.fields:
        targets[field.name] = fields[field.name].mask(is_event_masked, params=params)

    ablations = []
    for ablation in ablations:
        fields[ablation].ablate()

    targets = TensorDict(targets, batch_size=1)

    return PretrainingData.new(inputs=fields, targets=targets, meta=tuple(meta))


def stream(
    datapath: str,
    mode: str,
    params: Hyperparameters,
    manager: SystemManager,
    split: str,
) -> datapipes.iter.IterDataPipe:
    assert mode in ["pretrain", "finetune", "inference"]
    assert split in ["train", "validate", "test"]

    processor = ContextProcessor(manager=manager, params=params)

    return (
        datapipes.iter.FileLister(datapath, masks="*.avro")
        .shuffle()
        .sharding_filter()
        .flatmap(read)
        .filter(partial(subset, split=split))
        .shuffle()
        .flatmap(partial(sample, manager=manager, params=params, mode=mode, split=split))
        .map(processor.process)
        .shuffle()
        .map(partial(tensorfield, manager=manager, params=params))
        .map(partial(package, params=params, manager=manager, mode=mode))
    )


def collate(batch: list[SequenceData]) -> SequenceData:
    stacked = torch.stack(batch, dim=0).squeeze(1).auto_batch_size_(batch_dims=1)  # type: ignore
    stacked.meta = [observation.meta for observation in batch]

    return stacked


def mock(params: Hyperparameters, manager: SystemManager) -> TensorDict:
    output = {}

    N = params.batch_size
    L = params.n_context

    is_padded = torch.arange(L).expand(N, L).lt(torch.randint(1, L, [N]).unsqueeze(-1)).bool()

    tensorfield = dict(
        continuous=ContinuousField,
        temporal=TemporalField,
        discrete=DiscreteField,
        entity=EntityField,
    )

    for field in manager.schema.fields:
        if field.type in ["continuous", "temporal"]:
            indicators = torch.randint(0, len(Tokens), (N, L))
            padded = torch.where(is_padded, Tokens.PAD, indicators)
            is_valued = padded.eq(Tokens.VAL).long()
            values = torch.rand(N, L).mul(is_valued)
            output[field.name] = tensorfield[field.type](content=values, lookup=padded, batch_size=[N])  # type: ignore

        if field.type in ["discrete", "entity"]:
            limit = manager.levelsets.get_size(field) + len(Tokens) if field.type == "discrete" else L + len(Tokens)
            values = torch.randint(0, limit, (N, L))
            padded = torch.where(is_padded, Tokens.PAD, values)
            output[field.name] = tensorfield[field.type](lookup=padded, batch_size=[N])  # type: ignore

    return TensorDict(output, batch_size=N)
