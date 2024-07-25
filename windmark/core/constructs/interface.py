from functools import cache
from typing import Type, TypeAlias

import torch
from beartype.typing import Callable
from jaxtyping import Bool, Int
from tensordict.prototype import tensorclass

from windmark.core.constructs.general import FieldRequest, FieldType


@tensorclass
class TargetField:
    """
    Represents a target field in the windmark system.

    Attributes:
        lookup (Int[torch.Tensor, "_N L"]): The lookup value for the target field.
        is_masked (Bool[torch.Tensor, "_N L"]): Indicates whether the target field is masked or not.
    """

    lookup: Int[torch.Tensor, "_N L"]
    is_masked: Bool[torch.Tensor, "_N L"]


class TensorField:
    """
    Represents an abstract tensor field.

    Attributes:
        None

    Methods:
        new(cls, values: Any, field: FieldRequest, params: Hyperparameters, manager: SystemManager) -> "TensorField":
            Creates a new TensorField instance.

        mask(self, is_event_masked: torch.Tensor, params: Hyperparameters) -> "TargetField":
            Masks the tensor field based on the given mask tensor.

        prune(self) -> None:
            Prunes the tensor field.

        get_target_size(cls, params: Hyperparameters, manager: SystemManager, field: FieldRequest) -> int:
            Returns the target size of the tensor field.

        postprocess(cls, values: torch.Tensor, targets: torch.Tensor, params: Hyperparameters) -> torch.Tensor:
            Performs post-processing on the tensor field.

        mock(cls, field: FieldRequest, params: Hyperparameters, manager: SystemManager) -> "TensorField":
            Creates a mock TensorField instance.

    """

    # @classmethod
    # def new(cls, values: Any, field: FieldRequest, params: Hyperparameters, manager: SystemManager) -> "TensorField":
    #     pass

    # def mask(self, is_event_masked: torch.Tensor, params: Hyperparameters) -> "TargetField":
    #     pass

    # def prune(self) -> None:
    #     pass

    # @classmethod
    # def get_target_size(cls, params: Hyperparameters, manager: SystemManager, field: FieldRequest) -> int:
    #     pass

    # @classmethod
    # def postprocess(cls, values: torch.Tensor, targets: torch.Tensor, params: Hyperparameters) -> torch.Tensor:
    #     pass

    # @classmethod
    # def mock(cls, field: FieldRequest, params: Hyperparameters, manager: SystemManager) -> "TensorField":
    #     pass


class FieldEmbedder(torch.nn.Module):
    """Abstract template for a field embedder module"""

    # def __init__(self, params: Hyperparameters, manager: SystemManager, field: FieldRequest):
    #     pass

    # @jaxtyped(typechecker=beartype)
    # def forward(self, inputs: TensorField) -> Float[torch.Tensor, "_N L C"]:
    #     pass


Registrant: TypeAlias = Type[TensorField] | Type[FieldEmbedder]


class FieldInterface:
    """
    A class representing the interface for registering and accessing field types in the Windmark system.
    """

    tensorfields: dict[FieldType, Type[TensorField]] = {}
    embedders: dict[FieldType, Type[FieldEmbedder]] = {}

    @classmethod
    def register(cls, field: FieldType) -> Callable[[Registrant], Registrant]:
        """
        Decorator method for registering a field type.

        Args:
            field (FieldType): The field type to register.

        Returns:
            Callable[[Registrant], Registrant]: The decorator function.
        """
        if not isinstance(field, FieldType):
            raise KeyError("field key is not a registered field type")

        def decorator(registrant: Registrant) -> Registrant:
            """
            Decorator function used for registering a registrant object.

            Args:
                registrant (Registrant): The object to be registered.

            Returns:
                Registrant: The registered object.

            Raises:
                KeyError: If the registrant is already registered as a tensorfield or field embedder.
                ValueError: If the registrant is neither a tensorfield nor a field embedder.
            """
            registrant.type = field

            if issubclass(registrant, TensorField):
                if field in cls.tensorfields.keys():
                    raise KeyError("tensorfield is already registered")

                cls.tensorfields[field] = registrant

            elif issubclass(registrant, FieldEmbedder):
                if field in cls.embedders.keys():
                    raise KeyError("field embedder is already registered")

                cls.embedders[field] = registrant

            else:
                raise ValueError("object is neither an embedder or a tensorfield")

            return registrant

        return decorator

    @classmethod
    @cache
    def _validate(cls) -> None:
        """
        Internal method to validate the consistency between registered tensorfields and embedders.
        Raises an AttributeError if there are any inconsistencies.
        """
        tensorfields = set([tensorfield.name for tensorfield in cls.tensorfields.keys()])
        embedders = set([embedder.name for embedder in cls.embedders.keys()])

        if tensorfields != embedders:
            difference = tensorfields.symmetric_difference(embedders)

            raise AttributeError(f"interface differences: {difference}")

    @classmethod
    def tensorfield(cls, field: FieldRequest) -> Type[TensorField]:
        """
        Retrieves the tensorfield type associated with the given field request.

        Args:
            field (FieldRequest): The field request.

        Returns:
            Type[TensorField]: The tensorfield type.
        """
        cls._validate()

        return cls.tensorfields[field.type]

    @classmethod
    def embedder(cls, field: FieldRequest) -> Type[FieldEmbedder]:
        """
        Retrieves the field embedder type associated with the given field request.

        Args:
            field (FieldRequest): The field request.

        Returns:
            Type[FieldEmbedder]: The field embedder type.
        """
        cls._validate()

        return cls.embedders[field.type]
