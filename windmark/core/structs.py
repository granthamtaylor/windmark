import re
import functools
from enum import IntEnum
from typing import TypeAlias, Annotated
from dataclasses import dataclass

import numpy as np
import pydantic
import torch
from pytdigest import TDigest
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from tensordict import TensorDict
from tensordict.prototype import tensorclass
from torch import Tensor
from mashumaro.mixins.json import DataClassJSONMixin


class Tokens(IntEnum):

    """Special token representations"""

    VAL = 0
    UNK = 1
    PAD = 2
    MASK = 3


@dataclass
class Field(DataClassJSONMixin):
    name: str
    type: str

    def __post_init__(self):
        types = [
            "continuous",
            "discrete",
            "entity",
            "temporal",
        ]

        assert self.type in types, f"field type must be one of {types}"

        assert re.match(r"^[a-z][a-z0-9_]*$", self.name), f"invalid field name {self.name}"


@dataclass
class LevelSet(DataClassJSONMixin):
    name: str
    levels: list[str]
    is_valid: bool

    @classmethod
    def empty(cls, name: str) -> "LevelSet":
        return cls(name=name, levels=[], is_valid=False)

    @classmethod
    def from_levels(cls, name: str, levels: list[str]) -> "LevelSet":
        return cls(name=name, levels=levels, is_valid=True)

    @functools.cached_property
    def mapping(self) -> dict[str, int]:
        mapping = {level: index + len(Tokens) for index, level in enumerate(self.levels)}
        mapping["[UNK]"] = int(Tokens.UNK)

        return mapping


@dataclass
class Centroid(DataClassJSONMixin):
    name: str
    array: list[list[float]]
    is_valid: bool

    @classmethod
    def empty(cls, name: str) -> "Centroid":
        return cls(name=name, array=[], is_valid=False)

    @classmethod
    def from_digest(cls, name: str, digest: TDigest) -> "Centroid":
        array = digest.get_centroids().tolist()
        return cls(name=name, array=array, is_valid=True)


@pydantic.dataclasses.dataclass
class Hyperparameters(DataClassJSONMixin):
    # architectural
    batch_size: Annotated[int, pydantic.Field(gt=0, le=2048)] = 128
    """Batch size for training (how many observations per step)"""
    n_context: Annotated[int, pydantic.Field(gt=0, le=2048)] = 128
    """Context size (how many events per observation)"""
    d_field: Annotated[int, pydantic.Field(gt=1, le=256)] = 64
    """Hidden dimension per field"""
    n_heads_field_encoder: Annotated[int, pydantic.Field(gt=0, le=32)] = 4
    """Number of heads in field encoder"""
    n_layers_field_encoder: Annotated[int, pydantic.Field(gt=0, le=32)] = 2
    """Number of layers in field encoder"""
    n_heads_event_encoder: Annotated[int, pydantic.Field(gt=0, le=32)] = 8
    """Number of heads in event encoder"""
    n_layers_event_encoder: Annotated[int, pydantic.Field(gt=0, le=32)] = 8
    """Number of layers in event encoder"""
    dropout: Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] = 0.1
    """Dropout rate"""
    n_bands: Annotated[int, pydantic.Field(gt=1, le=512)] = 8
    """Precision of fourier feature encoders"""
    head_shape_log_base: Annotated[int, pydantic.Field(gt=1, le=8)] = 4
    """How quickly to converge sequence representation"""
    n_quantiles: Annotated[int, pydantic.Field(gt=0, lt=513)] = 64
    """Number of quantiles for continuous and temporal field"""

    # training
    n_steps: Annotated[int, pydantic.Field(gt=0)] = 128
    """Proportion of events to sample from during finetuning"""
    weight_decay: Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] = 0.001
    """Optimizer weight decay"""
    gradient_clip_val: Annotated[float, pydantic.Field(ge=0.0)] = 0.05
    """Gradient clipping threshold"""
    max_epochs: Annotated[int, pydantic.Field(gt=0, lt=257)] = 256
    """Maximum number of epochs for pretraining and finetuning"""
    quantile_smoothing: Annotated[float, pydantic.Field(gt=0.0, lt=33.0)] = 1.0
    """Smoothing factor of continuous fields' self-supervised"""
    p_mask_event: Annotated[float, pydantic.Field(gt=0.0, lt=1.0)] = 0.05
    """Probability of masking any event"""
    p_mask_field: Annotated[float, pydantic.Field(gt=0.0, lt=1.0)] = 0.05
    """Probability of masking any field"""
    n_epochs_frozen: Annotated[int, pydantic.Field(gt=0, le=128)] = 8
    """Number of epochs to freeze encoder while finetuning"""
    interpolation_rate: Annotated[float, pydantic.Field(gt=0.0, lt=1.0)] = 0.125
    """Interpolation rate of imbalanced classification labels"""
    learning_rate: Annotated[float, pydantic.Field(gt=0.0, lt=1.0)] = 0.0001
    """Learning rate"""

    @pydantic.model_validator(mode="after")
    def check_head_shape(self):
        assert self.d_field % self.n_heads_field_encoder == 0, "d_field must be divisible by n_heads_field_encoder"
        assert self.d_field % self.n_heads_event_encoder == 0, "d_field must be divisible by n_heads_event_encoder"

        return self

    @pydantic.model_validator(mode="after")
    def check_finetuning_unfreeze(self):
        assert self.n_epochs_frozen < self.max_epochs, "n_epochs_frozen must be less than max_epochs"

        return self


@tensorclass
class TargetField:
    lookup: Int[Tensor, "N L"]
    is_masked: Bool[Tensor, "N L"]


@tensorclass
class DiscreteField:
    lookup: Int[Tensor, "N L"]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(cls, values: list[int], params: Hyperparameters) -> "DiscreteField":
        padding = (params.n_context - len(values), 0)

        array = np.array(values, dtype=int)
        lookup = torch.nn.functional.pad(torch.tensor(array), pad=padding, value=Tokens.PAD).unsqueeze(0)

        return cls(lookup=lookup, batch_size=[1])

    @jaxtyped(typechecker=beartype)
    def mask(self, is_event_masked: Tensor, params: Hyperparameters) -> TargetField:
        N, L = (1, params.n_context)
        mask_token = torch.full((N, L), Tokens.MASK)

        is_field_masked = torch.rand(N, L).lt(params.p_mask_field)
        is_masked = is_field_masked.logical_or(is_event_masked)

        targets = self.lookup.clone()

        self.lookup = self.lookup.masked_scatter(is_masked, mask_token)

        return TargetField(lookup=targets, is_masked=is_masked, batch_size=self.batch_size)


@tensorclass
class EntityField:
    lookup: Int[Tensor, "N L"]

    new = DiscreteField.new
    mask = DiscreteField.mask


@tensorclass
class ContinuousField:
    lookup: Int[Tensor, "N L"]
    content: Float[Tensor, "N L"]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(cls, values, params: Hyperparameters) -> "ContinuousField":
        padding = (params.n_context - len(values), 0)

        values = np.nan_to_num(np.array(values, dtype=float))
        lookup = np.where(np.isnan(values), Tokens.PAD, Tokens.VAL)

        # this effectively creates `1-(1/inf)` to prevent an index error
        # somewhere in the dataset this exists a CDF of `1.0`, which will not be "floored" correctly
        dampener = 1 - torch.finfo(torch.half).tiny

        return cls(
            content=torch.nn.functional.pad(torch.tensor(values), pad=padding, value=0.0)
            .float()
            .unsqueeze(0)
            .mul(dampener),
            lookup=torch.nn.functional.pad(torch.tensor(lookup), pad=padding, value=Tokens.PAD).unsqueeze(0),
            batch_size=[1],
        )

    @jaxtyped(typechecker=beartype)
    def mask(self, is_event_masked: Tensor, params: Hyperparameters) -> TargetField:
        N, L = (1, params.n_context)
        mask_token = torch.full((N, L), Tokens.MASK)

        # fine out what to mask
        is_field_masked = torch.rand(N, L).lt(params.p_mask_field)
        is_masked = is_field_masked | is_event_masked

        # creating discrete targets
        quantiles = self.content.mul(params.n_quantiles).floor().long().add(len(Tokens))
        is_not_valued = self.lookup != 0
        targets = quantiles.masked_scatter(is_not_valued, quantiles)

        # mask original values
        self.lookup = self.lookup.masked_scatter(is_masked, mask_token)
        self.content *= ~is_masked

        # return SSL target
        return TargetField(lookup=targets, is_masked=is_masked, batch_size=self.batch_size)


@tensorclass
class TemporalField:
    lookup: Int[Tensor, "N L"]
    content: Float[Tensor, "N L"]

    new = ContinuousField.new
    mask = ContinuousField.mask


TensorField: TypeAlias = ContinuousField | DiscreteField | EntityField | TemporalField


@tensorclass
class PretrainingData:
    inputs: TensorDict[TensorField]
    targets: TensorDict[TargetField]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(cls, batch: tuple[TensorDict, TensorDict, tuple[str, str]]):
        inputs, targets, _ = batch

        return cls(inputs=inputs, targets=targets, batch_size=[1])


@tensorclass
class FinetuningData:
    inputs: TensorDict[TensorField]
    targets: Int[Tensor, "N T"]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(cls, batch: tuple[TensorDict, Tensor, tuple[str, str]]):
        inputs, targets, _ = batch

        targets = targets.unsqueeze(0)

        return cls(inputs=inputs, targets=targets, batch_size=[1])


@tensorclass
class InferenceData:
    inputs: TensorDict[TensorField]
    meta: list[tuple[str, str]] | tuple[str, str]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(cls, batch: tuple[TensorDict, Tensor, tuple[str, str]]):
        inputs, _, meta = batch

        return cls(inputs=inputs, meta=meta, batch_size=[1])


SequenceData: TypeAlias = PretrainingData | FinetuningData | InferenceData
