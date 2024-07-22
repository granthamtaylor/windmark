from functools import cache
from typing import Callable, Type, Any

import torch
from jaxtyping import Bool, Int
from tensordict.prototype import tensorclass

from windmark.core.constructs.general import FieldRequest, FieldType, Hyperparameters
from windmark.core.managers import SystemManager


@tensorclass
class TargetField:
    lookup: Int[torch.Tensor, "_N L"]
    is_masked: Bool[torch.Tensor, "_N L"]


class TensorField:
    @classmethod
    def new(cls, values: Any, field: FieldRequest, params: Hyperparameters, manager: SystemManager) -> "TensorField":
        pass

    def mask(self, is_event_masked: torch.Tensor, params: Hyperparameters) -> "TargetField":
        pass

    def prune(self) -> None:
        pass

    @classmethod
    def get_target_size(cls, params: Hyperparameters, manager: SystemManager, field: FieldRequest) -> int:
        pass

    @classmethod
    def postprocess(cls, values: torch.Tensor, targets: torch.Tensor, params: Hyperparameters) -> torch.Tensor:
        pass

    @classmethod
    def mock(cls, field: FieldRequest, params: Hyperparameters, manager: SystemManager) -> "TensorField":
        pass


class FieldEmbedder(torch.nn.Module):
    pass


class FieldInterface:
    tensorfields: dict[FieldType, Type[TensorField]] = {}
    embedders: dict[FieldType, Type[FieldEmbedder]] = {}

    @classmethod
    def register(cls, field: FieldType) -> Callable:
        if not isinstance(field, FieldType):
            raise KeyError("field key is not a registered field type")

        def decorator(registrant: Type[TensorField] | Type[FieldEmbedder]) -> Type[TensorField] | Type[FieldEmbedder]:
            registrant.type = field

            if issubclass(registrant, TensorField):
                if field in cls.tensorfields.keys():
                    raise KeyError("tensorfield is already registered")

                cls.tensorfields[field] = registrant

            elif issubclass(registrant, FieldEmbedder):
                if field in cls.embedders.keys():
                    raise KeyError("embedder is already registered")

                cls.embedders[field] = registrant

            else:
                raise ValueError("object is neither an embedder or a tensorfield")

            return registrant

        return decorator

    @classmethod
    @cache
    def _validate(cls) -> None:
        tensorfields = set([tensorfield.name for tensorfield in cls.tensorfields.keys()])
        embedders = set([embedder.name for embedder in cls.embedders.keys()])

        if tensorfields != embedders:
            difference = tensorfields.symmetric_difference(embedders)

            raise AttributeError(f"interface differences: {difference}")

    @classmethod
    def tensorfield(cls, field: FieldRequest) -> Type[TensorField]:
        cls._validate()

        return cls.tensorfields[field.type]

    @classmethod
    def embedder(cls, field: FieldRequest) -> Type[FieldEmbedder]:
        cls._validate()

        return cls.embedders[field.type]
