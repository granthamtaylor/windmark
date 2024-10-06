from typing import Annotated
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from tensordict import TensorDict
from tensordict import tensorclass
import torch

from windmark.core.constructs.tensorfields import TensorField, TargetField


class SequenceData:
    """
    Represents a sequential data package.
    """

    pass


@tensorclass
class PretrainingData(SequenceData):
    """
    A class representing pretraining data for a machine learning model.

    Attributes:
        inputs (Annotated[TensorDict, TensorField]): The input data for pretraining.
        targets (Annotated[TensorDict, TargetField]): The target data for pretraining.
        meta (list[tuple[str, str]] | tuple[str, str]): Additional metadata for the pretraining data.
    """

    inputs: Annotated[TensorDict, TensorField]
    targets: Annotated[TensorDict, TargetField]
    meta: list[tuple[str, str]] | tuple[str, str]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(cls, inputs: TensorDict, targets: TensorDict, meta: tuple[str, ...]):
        """
        Create a new instance of the PretrainingData class.

        Args:
            inputs (TensorDict): The input data for pretraining.
            targets (TensorDict): The target data for pretraining.
            meta (tuple[str, ...]): Additional metadata for the pretraining data.

        Returns:
            PretrainingData: A new instance of the PretrainingData class.
        """
        return cls(inputs=inputs, targets=targets, meta=meta, batch_size=[1])


@tensorclass
class SupervisedData(SequenceData):
    """
    A class representing supervised data for machine learning tasks.

    Attributes:
        inputs (TensorDict): The input data for the model.
        targets (torch.Tensor): The target data for the model.
        meta (list[tuple[str, str]] | tuple[str, str]): Additional metadata for the data.
    """

    inputs: Annotated[TensorDict, TensorField]
    targets: Int[torch.Tensor, "_N T"]
    meta: list[tuple[str, str]] | tuple[str, str]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(cls, inputs: TensorDict, targets: torch.Tensor, meta: tuple[str, ...]):
        """
        Create a new instance of the class.

        Args:
            inputs (TensorDict): The input tensors.
            targets (torch.Tensor): The target tensor.
            meta (tuple[str, ...]): The metadata.

        Returns:
            cls: The new instance of the class.
        """
        targets = targets.unsqueeze(0)

        return cls(inputs=inputs, targets=targets, meta=meta, batch_size=[1])


@tensorclass
class OutputData:
    """
    Represents the output data of a model.

    Attributes:
        sequence (Float[torch.Tensor, "_N FdC"]): The sequence data.
        decoded_events (TensorDict): The decoded events.
        decoded_static_fields (TensorDict): The decoded static fields.
        predictions (Float[torch.Tensor, "_N T"]): The predictions.
    """

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
        """
        Create a new instance of the class.

        Args:
            cls: The class itself.
            sequence: The sequence of floats.
            decoded_events: A dictionary of decoded events.
            decoded_static_fields: A dictionary of decoded static fields.
            predictions: The predictions.

        Returns:
            An instance of the class with the given arguments.
        """
        assert sequence.shape[0] == decoded_events.shape[0] == decoded_static_fields.shape[0] == predictions.shape[0]

        L = sequence.shape[0]

        return cls(
            sequence=sequence,
            decoded_events=decoded_events,
            decoded_static_fields=decoded_static_fields,
            predictions=predictions,
            batch_size=[L],
        )
