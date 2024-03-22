import os
import random
from functools import partial

import fastavro
import numpy as np
import torch
from pytdigest import TDigest
from tensordict import TensorDict
from torchdata import datapipes

from windmark.core.managers import ClassificationManager
from windmark.core.structs import (
    ContinuousField,
    DiscreteField,
    EntityField,
    Field,
    FinetuningData,
    Hyperparameters,
    InferenceData,
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


def sample(
    sequence: dict,
    params: Hyperparameters,
    fields: list[Field],
    balancer: ClassificationManager,
    mode: str,
) -> list[dict[str, int | float | None]]:
    observations: list[dict[str, int | float | None]] = []

    for event in range(sequence["size"]):
        if mode == "pretrain":
            if params.pretrain_sample_rate < random.random():
                continue

            label = -1

        elif mode == "finetune":
            label = sequence["target"][event]

            if (label is None) or (label == -1):
                continue

            if params.finetune_sample_rate < random.random():
                continue

            if balancer.thresholds[label] < random.random():
                continue

        elif mode == "inference":
            label = -1

        window = slice(max(0, event - params.n_context), event)

        observation = dict(
            sequence_id=str(sequence["sequence_id"]),
            event_id=str(sequence["event_ids"][event]),
            label=label,
        )

        for field in fields:
            observation[field.name] = sequence[field.name][window]

        observations.append(observation)

    return observations


def hash(
    observation: dict[str, str | list[int] | list[float | None] | list[str]],
    fields: list[Field],
    params: Hyperparameters,
) -> dict[str, list[int] | list[float | None]]:
    offset = len(Tokens)

    for field in fields:
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
    fields: list[Field],
    digests: dict[str, TDigest],
) -> dict[str, list[int] | np.ndarray]:
    for field in fields:
        if field.type in ["continuous", "temporal"]:
            digest: TDigest = digests[field.name]
            array = np.array(observation[field.name], dtype=np.float64)
            observation[field.name] = digest.cdf(array)

    return observation


def tensorfield(
    observation: dict[str, list[int] | np.ndarray], params: Hyperparameters, fields: list[Field]
) -> tuple[TensorDict, torch.Tensor, tuple[str, str]]:
    output = {}

    tensorclasses = dict(
        discrete=DiscreteField,
        continuous=ContinuousField,
        entity=EntityField,
        temporal=TemporalField,
    )

    for field in fields:
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
    fields: list[Field],
) -> tuple[TensorDict, TensorDict, tuple[str, str]]:
    inputs, _, meta = observation

    N, L = (1, params.n_context)

    targets = {}

    is_event_masked = torch.rand(N, L).lt(params.p_mask_event)

    for field in fields:
        targets[field.name] = inputs[field.name].mask(is_event_masked, params=params)

    targets = TensorDict(targets, batch_size=1)

    return inputs, targets, meta


def package(
    observation: tuple[TensorDict, torch.Tensor, tuple[str, str]],
    params: Hyperparameters,
    fields: list[Field],
    mode: str,
) -> SequenceData:
    if mode == "pretrain":
        observation: tuple[TensorDict, TensorDict, tuple[str, str]] = mask(
            observation=observation, params=params, fields=fields
        )

    tensorclasses = dict(
        pretrain=PretrainingData,
        finetune=FinetuningData,
        inference=InferenceData,
    )

    return tensorclasses[mode].new(observation)


def stream(
    datapath: str | os.PathLike,
    mode: str,
    masks: str,
    centroids: dict[str, np.ndarray],
    fields: list[Field],
    params: Hyperparameters,
    balancer: ClassificationManager,
) -> datapipes.iter.IterDataPipe:
    digests = {field: TDigest.of_centroids(centroid) for field, centroid in centroids.items()}

    assert mode in ["pretrain", "finetune", "inference"]

    print(f"creating {mode} datapipe")

    sampler = partial(sample, fields=fields, params=params, balancer=balancer, mode=mode)

    return (
        datapipes.iter.FileLister(datapath, masks=masks)
        .shuffle()
        .sharding_filter()
        .flatmap(read)
        .shuffle()
        .flatmap(sampler)
        .map(partial(cdf, fields=fields, digests=digests))
        .map(partial(hash, fields=fields, params=params))
        .shuffle()
        .map(partial(tensorfield, fields=fields, params=params))
        .map(partial(package, params=params, fields=fields, mode=mode))
    )


def collate(batch: list[SequenceData]) -> SequenceData:
    stacked = torch.stack(batch, dim=0).squeeze(1).auto_batch_size_(batch_dims=1)

    if isinstance(stacked, InferenceData):
        stacked.meta = [observation.meta for observation in batch]

    return stacked


def mock(params: Hyperparameters, fields: list[Field]) -> TensorDict:
    data = {}

    N = params.batch_size
    L = params.n_context

    is_padded = torch.arange(L).expand(N, L).lt(torch.randint(1, L, [N]).unsqueeze(-1)).bool()

    for field in fields:
        match field.type:
            case "continuous" | "temporal":
                indicators = torch.randint(0, len(Tokens), (N, L))
                padded = torch.where(is_padded, Tokens.PAD, indicators)
                is_empty = padded.eq(Tokens.VAL).long()
                values = torch.rand(N, L).mul(is_empty)
                data[field.name] = ContinuousField(content=values, lookup=padded, batch_size=[N])

            case "discrete":
                values = torch.randint(0, field.levels + len(Tokens), (N, L))
                padded = torch.where(is_padded, Tokens.PAD, values)
                data[field.name] = DiscreteField(lookup=padded, batch_size=[N])

            case "entity":
                values = torch.randint(0, L + len(Tokens), (N, L))
                padded = torch.where(is_padded, Tokens.PAD, values)
                data[field.name] = EntityField(lookup=padded, batch_size=[N])

    return TensorDict(data, batch_size=N)
