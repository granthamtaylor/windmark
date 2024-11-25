# Copyright Grantham Taylor.

from random import random
from typing import TypeAlias, Any, Iterator
from functools import partial
from enum import Enum

from windmark.core.constructs.managers import SystemManager
from windmark.core.constructs.general import Hyperparameters

AnnotationType: TypeAlias = tuple[str, str, int]
FieldType: TypeAlias = dict[str, list[Any] | Any]


def index(sequence: dict, manager: SystemManager) -> Iterator[int]:
    return range(len(sequence[manager.schema.event_id]))


def pretrain(sequence: dict, params: Hyperparameters, manager: SystemManager, split: str) -> Iterator[tuple[int, int]]:
    events = index(sequence, manager=manager)
    rate = manager.sample.pretraining[split]
    out = ((event, -1) for event in events if rate > random())

    return out


def finetune(sequence: dict, params: Hyperparameters, manager: SystemManager, split: str) -> Iterator[tuple[int, int]]:
    events = index(sequence, manager=manager)
    labels = sequence[manager.schema.target_id]
    mapping = manager.task.balancer.label_mapping
    split_rate = manager.sample.finetuning[split]
    rates: dict[str, float] = manager.task.balancer.sample_rates_mapping

    out = (
        (event, mapping[label])
        for event, label in zip(events, labels)
        if (label is not None) & ((rates[label] * split_rate > random()) or (split == "test"))
    )

    return out


def inference(sequence: dict, params: Hyperparameters, manager: SystemManager, split: str) -> Iterator[tuple[int, int]]:
    mapping = manager.task.balancer.mapping

    if params.predict_only_sequence_end:
        event: int = len(sequence[manager.schema.event_id]) - 1
        target: int = mapping[sequence[manager.schema.target_id][-1]]
        return [(event, target)]

    targets = (mapping[label] for label in sequence[manager.schema.target_id])
    zipped = zip(index(sequence, manager=manager), targets)
    out = ((event, target) for event, target in zipped)

    return out


class Sampler(Enum):
    pretrain = partial(pretrain)
    finetune = partial(finetune)
    inference = partial(inference)


def sample(
    sequence: dict,
    params: Hyperparameters,
    manager: SystemManager,
    split: str,
    sampler: Sampler,
) -> Iterator[tuple[AnnotationType, FieldType]]:
    indices: list[tuple[int, int]] = sampler(sequence=sequence, params=params, manager=manager, split=split)

    for event, target in indices:
        window = slice(max(0, event - params.n_context), event)

        sequence_id = str(sequence[manager.schema.sequence_id])
        event_id = str(sequence[manager.schema.event_id][event])

        annotations: AnnotationType = (sequence_id, event_id, target)

        fields = {}

        for field in manager.schema.dynamic:
            assert len(sequence[manager.schema.event_id]) == len(sequence[field.name])
            fields[field.name] = sequence[field.name][window]

        for field in manager.schema.static:
            fields[field.name] = sequence[field.name]

        yield annotations, fields
