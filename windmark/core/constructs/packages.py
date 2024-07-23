from typing import Annotated
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from tensordict import TensorDict
from tensordict.prototype import tensorclass
import torch

from windmark.core.constructs.tensorfields import TensorField, TargetField


class SequenceData:
    pass


@tensorclass
class PretrainingData(SequenceData):
    inputs: Annotated[TensorDict, TensorField]
    targets: Annotated[TensorDict, TargetField]
    meta: list[tuple[str, str]] | tuple[str, str]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(cls, inputs: TensorDict, targets: TensorDict, meta: tuple[str, ...]):
        return cls(inputs=inputs, targets=targets, meta=meta, batch_size=[1])


@tensorclass
class SupervisedData(SequenceData):
    inputs: Annotated[TensorDict, TensorField]
    targets: Int[torch.Tensor, "_N T"]
    meta: list[tuple[str, str]] | tuple[str, str]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(cls, inputs: TensorDict, targets: torch.Tensor, meta: tuple[str, ...]):
        targets = targets.unsqueeze(0)

        return cls(inputs=inputs, targets=targets, meta=meta, batch_size=[1])


@tensorclass
class OutputData:
    sequence: Float[torch.Tensor, "_N FdC"]
    decoded_events: TensorDict
    decoded_static_fields: TensorDict
    predictions: Float[torch.Tensor, "_N T"]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(
        cls,
        sequence: Float[torch.Tensor, "_N FdC"],
        decoded_events: TensorDict,
        decoded_static_fields: TensorDict,
        predictions: Float[torch.Tensor, "_N T"],
    ):
        assert sequence.shape[0] == decoded_events.shape[0] == decoded_static_fields.shape[0] == predictions.shape[0]

        L = sequence.shape[0]

        return cls(
            sequence=sequence,
            decoded_events=decoded_events,
            decoded_static_fields=decoded_static_fields,
            predictions=predictions,
            batch_size=[L],
        )
