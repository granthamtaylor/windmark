import random
from functools import partial
from typing import TypeAlias, Any

import fastavro
import torch
from tensordict import TensorDict
from torchdata import datapipes

from windmark.core.managers import SystemManager
from windmark.core.constructs.general import Hyperparameters
from windmark.core.constructs.interface import FieldInterface
from windmark.core.constructs.packages import SupervisedData, PretrainingData, SequenceData


AnnotationType: TypeAlias = tuple[str, str, int]
FieldType: TypeAlias = dict[str, list[Any]]


def read(filename: str) -> list[dict[str, Any]]:
    with open(filename, "rb") as f:
        reader = fastavro.reader(f)
        records = [record for record in reader]

    return records


def subset(sequence: dict[str, Any], split: str) -> bool:
    return sequence["split"] == split


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


def tensorfield(
    observation: tuple[AnnotationType, FieldType],
    params: Hyperparameters,
    manager: SystemManager,
) -> tuple[AnnotationType, TensorDict]:
    annotations, fields = observation

    output = {}

    for field in manager.schema.fields:
        values = fields[field.name]
        tensorfield = FieldInterface.tensorfield(field)
        output[field.name] = tensorfield.new(values=values, field=field, params=params, manager=manager)

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

    pruned_fields = []
    for pruned_field in pruned_fields:
        assert pruned_field in fields.keys(), f'pruned field "{pruned_field}" not found'
        fields[pruned_field].ablate()

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

    return (
        datapipes.iter.FileLister(datapath, masks="*.avro")
        .shuffle()
        .sharding_filter()
        .flatmap(read)
        .filter(partial(subset, split=split))
        .shuffle()
        .flatmap(partial(sample, manager=manager, params=params, mode=mode, split=split))
        .shuffle()
        .map(partial(tensorfield, manager=manager, params=params))
        .map(partial(package, params=params, manager=manager, mode=mode))
    )


def mock(params: Hyperparameters, manager: SystemManager) -> TensorDict:
    observations = []

    for _ in range(params.batch_size):
        tree = {}

        for field in manager.schema.fields:
            tensorfield = FieldInterface.tensorfield(field)
            tree[field.name] = tensorfield.mock(field=field, params=params, manager=manager)

        observations.append(TensorDict(tree, batch_size=[1]))

    return torch.stack(observations, dim=0).squeeze(1)


def collate(batch: list[SequenceData]) -> SequenceData:
    stacked = torch.stack(batch, dim=0).squeeze(1).auto_batch_size_(batch_dims=1)  # type: ignore
    stacked.meta = [observation.meta for observation in batch]

    return stacked
