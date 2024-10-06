import random
from typing import TypeAlias, Any, Generator, Callable
from types import SimpleNamespace

from windmark.core.managers import SystemManager
from windmark.core.constructs.general import Hyperparameters

AnnotationType: TypeAlias = tuple[str, str, int]
FieldType: TypeAlias = dict[str, list[Any] | Any]


def index(sequence: dict, manager: SystemManager) -> Generator[int, None, None]:
    return range(len(sequence[manager.schema.event_id]))


def pretrain(sequence: dict, params: Hyperparameters, manager: SystemManager, split: str) -> list[tuple[int, int]]:
    events = index(sequence)

    rate = manager.sample.pretraining[split]

    events = [event for event in events if rate > random.random()]
    targets = [-1] * len(events)

    return zip(events, targets)


def finetune(sequence: dict, params: Hyperparameters, manager: SystemManager, split: str) -> list[tuple[int, int]]:
    events = index(sequence)

    thresholds: dict[str, float] = manager.task.balancer.thresholds

    split_rate = manager.sample.finetuning[split]

    rates = {target: threshold * split_rate for target, threshold in thresholds.items()}

    out: list[tuple[int, int]] = []

    for event in events:
        label: str | None = sequence[manager.schema.target_id][event]

        if label is None:
            continue
        else:
            target: int = manager.task.balancer.mapping[label]

        if (split != "test") & (rates[target] < random.random()):
            continue

        out.append((event, target))

    return out


def inference(sequence: dict, params: Hyperparameters, manager: SystemManager, split: str) -> list[tuple[int, int]]:
    if params.predict_only_sequence_end:
        # FIXME minus one, right??
        event = [len(sequence[manager.schema.event_id]) - 1]
        target = [sequence[manager.schema.target_id][-1]]

        return event, target
        # return [len(sequence[manager.schema.event_id])]

    events = index(sequence)
    targets = [target for target in sequence[manager.schema.target_id]]

    return zip(list(events), targets)


samplers = SimpleNamespace(
    pretrain=pretrain,
    finetune=finetune,
    inference=inference,
)


def sample(
    sequence: dict,
    params: Hyperparameters,
    manager: SystemManager,
    split: str,
    sampler: Callable,
) -> Generator[tuple[AnnotationType, FieldType], None, None]:
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
