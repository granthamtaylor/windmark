from typing import TypeAlias, Annotated
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from tensordict import TensorDict
from tensordict.prototype import tensorclass
from torch import Tensor

from windmark.core.constructs.tensorfields import TensorField, TargetField


@tensorclass
class PretrainingData:
    inputs: Annotated[TensorDict, TensorField]
    targets: Annotated[TensorDict, TargetField]
    meta: list[tuple[str, str]] | tuple[str, str]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(cls, inputs: TensorDict, targets: TensorDict, meta: tuple[str, ...]):
        return cls(inputs=inputs, targets=targets, meta=meta, batch_size=[1])


@tensorclass
class SupervisedData:
    inputs: Annotated[TensorDict, TensorField]
    targets: Int[Tensor, "N T"]
    meta: list[tuple[str, str]] | tuple[str, str]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(cls, inputs: TensorDict, targets: Tensor, meta: tuple[str, ...]):
        targets = targets.unsqueeze(0)

        return cls(inputs=inputs, targets=targets, meta=meta, batch_size=[1])


SequenceData: TypeAlias = PretrainingData | SupervisedData


@tensorclass
class OutputData:
    sequence: Float[Tensor, "N FC"]
    reconstructions: TensorDict
    predictions: Float[Tensor, "N T"]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(
        cls,
        sequence: Float[Tensor, "N FC"],
        reconstructions: TensorDict,
        predictions: Float[Tensor, "N T"],
    ):
        assert sequence.shape[0] == reconstructions.shape[0] == predictions.shape[0]

        L = sequence.shape[0]

        return cls(sequence=sequence, reconstructions=reconstructions, predictions=predictions, batch_size=[L])
