from typing import TypeVar

from windmark.core.constructs.general import FieldRequest
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
    modules: dict[str, dict[str, TypeVar]] = {
        "continuous": {"tensorfield": ContinuousField, "embedder": ContinuousFieldEmbedder},
        "discrete": {"tensorfield": DiscreteField, "embedder": DiscreteFieldEmbedder},
        "static_continuous": {"tensorfield": StaticContinuousField, "embedder": StaticContinuousFieldEmbedder},
        "static_discrete": {"tensorfield": StaticDiscreteField, "embedder": StaticDiscreteFieldEmbedder},
        "entity": {"tensorfield": EntityField, "embedder": EntityFieldEmbedder},
        "temporal": {"tensorfield": TemporalField, "embedder": TemporalFieldEmbedder},
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
