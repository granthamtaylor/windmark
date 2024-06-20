from typing import TypeVar

from windmark.core.constructs.general import FieldRequest, FieldType
from windmark.core.architecture.embedders import (
    DiscreteFieldEmbedder,
    EntityFieldEmbedder,
    ContinuousFieldEmbedder,
    TemporalFieldEmbedder,
    StaticDiscreteFieldEmbedder,
    StaticContinuousFieldEmbedder,
)
from windmark.core.constructs.tensorfields import (
    DiscreteField,
    EntityField,
    ContinuousField,
    TemporalField,
    StaticDiscreteField,
    StaticContinuousField,
)


class FieldInterface:
    modules: dict[FieldType, dict[str, TypeVar]] = {
        FieldType.Numbers: {"tensorfield": ContinuousField, "embedder": ContinuousFieldEmbedder},
        FieldType.Categories: {"tensorfield": DiscreteField, "embedder": DiscreteFieldEmbedder},
        FieldType.Number: {"tensorfield": StaticContinuousField, "embedder": StaticContinuousFieldEmbedder},
        FieldType.Category: {"tensorfield": StaticDiscreteField, "embedder": StaticDiscreteFieldEmbedder},
        FieldType.Entities: {"tensorfield": EntityField, "embedder": EntityFieldEmbedder},
        FieldType.Timestamps: {"tensorfield": TemporalField, "embedder": TemporalFieldEmbedder},
    }

    @classmethod
    def check(cls, field: FieldRequest) -> None:
        assert field.type in list(cls.modules.keys())

    @classmethod
    def tensorfield(cls, field: FieldRequest) -> TypeVar:
        return cls.modules[field.type]["tensorfield"]

    @classmethod
    def embedder(cls, field: FieldRequest) -> TypeVar:
        return cls.modules[field.type]["embedder"]
