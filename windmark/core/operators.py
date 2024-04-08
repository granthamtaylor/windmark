import random
from functools import partial

import fastavro
import numpy as np
import torch
from pytdigest import TDigest
from tensordict import TensorDict
from torchdata import datapipes

from windmark.core.managers import SystemManager
from windmark.core.structs import (
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


def sample(
    sequence: dict,
    params: Hyperparameters,
    manager: SystemManager,
    split: str,
    mode: str,
) -> list[dict[str, str | list[int] | list[float | None] | list[str]]]:
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
            label = -1

        else:
            raise ValueError

        window = slice(max(0, event - params.n_context), event)

        observation = dict(
            sequence_id=str(sequence["sequence_id"]),
            event_id=str(sequence["event_ids"][event]),
            label=label,
        )

        for field in manager.schema.fields:
            observation[field.name] = sequence[field.name][window]

        observations.append(observation)

    return observations


def tokenize(
    observation: dict[str, str | list[float | None] | list[str]],
    manager: SystemManager,
) -> dict[str, str | list[int] | list[float | None] | list[str]]:
    for field in manager.schema.fields:
        if field.type == "discrete":
            mapping = manager.levelsets[field]

            values: list[str] = observation[field.name]

            observation[field.name] = list(map(lambda value: mapping[value], values))

    return observation


def hash(
    observation: dict[str, str | list[int] | list[float | None] | list[str]],
    manager: SystemManager,
    params: Hyperparameters,
) -> dict[str, str | list[int] | list[float | None]]:
    offset = len(Tokens)

    for field in manager.schema.fields:
        if field.type == "entity":
            values: list[str] = observation[field.name]

            unique = set(values)

            integers = random.sample(range(offset, params.n_context + offset), len(unique))

            mapping = dict(zip(unique, integers))

            mapping.update({"[UNK]": Tokens.UNK})

            observation[field.name] = list(map(lambda value: mapping[value], values))

    return observation


def cdf(
    observation: dict[str, str | list[int] | list[float | None]],
    manager: SystemManager,
) -> dict[str, str | list[int] | np.ndarray]:
    for field in manager.schema.fields:
        if field.type in ["continuous", "temporal"]:
            # TODO verify that this is caching properly
            digest: TDigest = manager.centroids.digests[field.name]
            array = np.array(observation[field.name], dtype=np.float64)
            observation[field.name] = digest.cdf(array)

    return observation


def tensorfield(
    observation: dict[str, str | list[int] | np.ndarray],
    params: Hyperparameters,
    manager: SystemManager,
) -> tuple[TensorDict, torch.Tensor, tuple[str, str]]:
    output = {}

    tensorclasses = dict(
        discrete=DiscreteField,
        continuous=ContinuousField,
        entity=EntityField,
        temporal=TemporalField,
    )

    for field in manager.schema.fields:
        values = observation[field.name]
        tensorclass = tensorclasses[field.type]
        output[field.name] = tensorclass.new(values, params=params)

    inputs = TensorDict(output, batch_size=1)
    labels = torch.tensor(observation["label"])
    meta = observation["sequence_id"], observation["event_id"]

    return inputs, labels, meta


def mask(
    observation: tuple[TensorDict, torch.Tensor, tuple[str, str]],
    params: Hyperparameters,
    manager: SystemManager,
) -> tuple[TensorDict, TensorDict, tuple[str, str]]:
    inputs, _, meta = observation

    N, L = (1, params.n_context)

    targets = {}

    is_event_masked = torch.rand(N, L).lt(params.p_mask_event)

    for field in manager.schema.fields:
        targets[field.name] = inputs[field.name].mask(is_event_masked, params=params)

    targets = TensorDict(targets, batch_size=1)

    return inputs, targets, meta


def package(
    observation: tuple[TensorDict, torch.Tensor, tuple[str, str]],
    params: Hyperparameters,
    manager: SystemManager,
    mode: str,
) -> SequenceData:
    assert mode in ("pretrain", "finetune", "inference")

    if mode in ["finetune", "inference"]:
        return SupervisedData.new(observation)

    return PretrainingData.new(mask(observation=observation, params=params, manager=manager))


def stream(
    datapath: str,
    mode: str,
    params: Hyperparameters,
    manager: SystemManager,
    split: str,
) -> datapipes.iter.IterDataPipe:
    assert mode in ["pretrain", "finetune", "inference"]
    assert split in ["train", "validate", "test"]

    return (
        datapipes.iter.FileLister(datapath, masks="*.avro")
        .shuffle()
        .sharding_filter()
        .flatmap(read)
        .filter(partial(subset, split=split))
        .shuffle()
        .flatmap(partial(sample, manager=manager, params=params, mode=mode, split=split))
        .map(partial(tokenize, manager=manager))
        .map(partial(cdf, manager=manager))
        .map(partial(hash, manager=manager, params=params))
        .shuffle()
        .map(partial(tensorfield, manager=manager, params=params))
        .map(partial(package, params=params, manager=manager, mode=mode))
    )


def collate(batch: list[SequenceData]) -> SequenceData:
    stacked = torch.stack(batch, dim=0).squeeze(1).auto_batch_size_(batch_dims=1)

    if isinstance(batch[0], SupervisedData):
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
            output[field.name] = tensorfield[field.type](content=values, lookup=padded, batch_size=[N])

        if field.type in ["discrete", "entity"]:
            limit = manager.levelsets.get_size(field) + len(Tokens) if field.type == "discrete" else L + len(Tokens)
            values = torch.randint(0, limit, (N, L))
            padded = torch.where(is_padded, Tokens.PAD, values)
            output[field.name] = tensorfield[field.type](lookup=padded, batch_size=[N])

    return TensorDict(output, batch_size=N)
