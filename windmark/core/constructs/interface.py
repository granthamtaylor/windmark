from typing import TypeVar

from windmark.core.constructs.general import FieldRequest, FieldType
from windmark.core.architecture.embedders import (
    DiscreteFieldEmbedder,
    StaticDiscreteFieldEmbedder,
    ContinuousFieldEmbedder,
    StaticContinuousFieldEmbedder,
    QuantileFieldEmbedder,
    StaticQuantileFieldEmbedder,
    EntityFieldEmbedder,
    TemporalFieldEmbedder,
)
from windmark.core.constructs.tensorfields import (
    DiscreteField,
    StaticDiscreteField,
    ContinuousField,
    StaticContinuousField,
    QuantileField,
    StaticQuantileField,
    EntityField,
    TemporalField,
)


class FieldInterface:
    modules: dict[FieldType, dict[str, TypeVar]] = {
        FieldType.Numbers: {"tensorfield": ContinuousField, "embedder": ContinuousFieldEmbedder},
        FieldType.Number: {"tensorfield": StaticContinuousField, "embedder": StaticContinuousFieldEmbedder},
        FieldType.Categories: {"tensorfield": DiscreteField, "embedder": DiscreteFieldEmbedder},
        FieldType.Category: {"tensorfield": StaticDiscreteField, "embedder": StaticDiscreteFieldEmbedder},
        FieldType.Quantiles: {"tensorfield": QuantileField, "embedder": QuantileFieldEmbedder},
        FieldType.Quantile: {"tensorfield": StaticQuantileField, "embedder": StaticQuantileFieldEmbedder},
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
