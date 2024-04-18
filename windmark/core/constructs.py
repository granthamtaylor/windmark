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

    # FIXME need to add ablation token
    # ABLATE = 4


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
    n_layers_field_encoder: Annotated[int, pydantic.Field(gt=0, le=32)] = 1
    """Number of layers in field encoder"""
    n_heads_event_encoder: Annotated[int, pydantic.Field(gt=0, le=32)] = 8
    """Number of heads in event encoder"""
    n_layers_event_encoder: Annotated[int, pydantic.Field(gt=0, le=32)] = 8
    """Number of layers in event encoder"""
    dropout: Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] = 0.1
    """Dropout rate"""
    n_bands: Annotated[int, pydantic.Field(gt=1, le=16)] = 8
    """Precision of fourier feature encoders"""
    head_shape_log_base: Annotated[int, pydantic.Field(gt=1, le=8)] = 4
    """How quickly to converge sequence representation"""
    n_quantiles: Annotated[int, pydantic.Field(gt=1, le=512)] = 64
    """Number of quantiles for continuous and temporal field"""

    # training
    n_pretrain_steps: Annotated[int, pydantic.Field(gt=0)] = 128
    """Number of steps to take per epoch during pretraining"""
    n_finetune_steps: Annotated[int, pydantic.Field(gt=0)] = 128
    """Number of steps to take per epoch during finetuning"""
    swa_lr: Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] = 0.1e-2
    """Stochastic Weight Averaging"""
    gradient_clip_val: Annotated[float, pydantic.Field(gt=0.0)] = 0.25
    """Gradient clipping threshold"""
    max_pretrain_epochs: Annotated[int, pydantic.Field(gt=0, le=1028)] = 256
    """Maximum number of epochs for pretraining"""
    max_finetune_epochs: Annotated[int, pydantic.Field(gt=0, le=1028)] = 256
    """Maximum number of epochs for finetuning"""
    quantile_smoothing: Annotated[float, pydantic.Field(gt=0.0, lt=33.0)] = 1.0
    """Smoothing factor of continuous fields' quantile labels"""
    p_mask_event: Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] = 0.075
    """Probability of masking any event"""
    p_mask_field: Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] = 0.075
    """Probability of masking any field"""
    n_epochs_frozen: Annotated[int, pydantic.Field(gt=0, le=128)] = 8
    """Number of epochs to freeze encoder while finetuning"""
    interpolation_rate: Annotated[float, pydantic.Field(ge=0.0, le=1.0)] = 0.08
    """Interpolation rate of imbalanced classification labels"""
    learning_rate: Annotated[float, pydantic.Field(gt=0.0, lt=1.0)] = 0.0001
    """Learning Rate during Pretraining"""
    learning_rate_dampener: Annotated[float, pydantic.Field(gt=0.0, lt=1.0)] = 0.1
    """Learning Rate Modifier during Finetuning"""
    patience: Annotated[int, pydantic.Field(ge=1, le=256)] = 16
    """Number of Epochs Patience for Early Stopping"""

    @pydantic.model_validator(mode="after")
    def check_head_shape(self):
        assert self.d_field % self.n_heads_field_encoder == 0, "d_field must be divisible by n_heads_field_encoder"
        assert self.d_field % self.n_heads_event_encoder == 0, "d_field must be divisible by n_heads_event_encoder"

        return self

    @pydantic.model_validator(mode="after")
    def check_finetuning_unfreeze(self):
        assert self.n_epochs_frozen < self.max_finetune_epochs, "n_epochs_frozen must be less than max_finetune_epochs"

        return self

    @pydantic.model_validator(mode="after")
    def check_mask_rates(self):
        rates = [self.p_mask_event, self.p_mask_field, self.p_mask_field + self.p_mask_event]
        assert max(rates) >= 0.01, "the masking rates are too low for any meaningful pretraining"

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

        return TargetField(lookup=targets, is_masked=is_masked, batch_size=self.batch_size)  # type: ignore

    def ablate(self):
        self.lookup = torch.full_like(self.lookup, Tokens.UNK)


@tensorclass
class EntityField:
    lookup: Int[Tensor, "N L"]

    new = DiscreteField.new
    mask = DiscreteField.mask
    ablate = DiscreteField.ablate


@tensorclass
class ContinuousField:
    lookup: Int[Tensor, "N L"]
    content: Float[Tensor, "N L"]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(cls, values, params: Hyperparameters) -> "ContinuousField":
        lookup = np.where(np.isnan(values), Tokens.UNK, Tokens.VAL)
        values = np.nan_to_num(np.array(values, dtype=float))

        padding = (params.n_context - len(values), 0)

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
        return TargetField(lookup=targets, is_masked=is_masked, batch_size=self.batch_size)  # type: ignore

    def ablate(self):
        self.lookup = torch.full_like(self.lookup, Tokens.UNK)
        self.content = torch.zeros_like(self.content)


@tensorclass
class TemporalField:
    lookup: Int[Tensor, "N L"]
    content: Float[Tensor, "N L"]

    new = ContinuousField.new
    mask = ContinuousField.mask
    ablate = ContinuousField.ablate


@tensorclass
class NeoTemporalField:
    lookup: Int[Tensor, "N L"]
    day_of_week: Int[Tensor, "N L"]
    day_of_year: Float[Tensor, "N L"]
    time_of_day: Float[Tensor, "N L"]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(cls, values, params: Hyperparameters) -> "NeoTemporalField":
        padding = (params.n_context - len(values), 0)

        lookup = np.where(np.isnan(values), Tokens.UNK, Tokens.VAL)
        day_of_week = np.where(np.isnan(((values.view("int64") - 4) % 7) + len(Tokens)), Tokens.UNK, Tokens.VAL)
        day_of_year = np.nan_to_num(values - values.astype("datetime64[Y]")) * (1 / 366)
        time_of_day = np.nan_to_num((values - values.astype("datetime64[D]")).astype(int)) * (1 / 1440)

        return cls(
            time_of_day=torch.nn.functional.pad(torch.tensor(time_of_day), pad=padding, value=0.0).unsqueeze(0),
            day_of_year=torch.nn.functional.pad(torch.tensor(day_of_year), pad=padding, value=Tokens.PAD).unsqueeze(0),
            day_of_week=torch.nn.functional.pad(torch.tensor(day_of_week), pad=padding, value=Tokens.PAD).unsqueeze(0),
            lookup=torch.nn.functional.pad(torch.tensor(lookup), pad=padding, value=Tokens.PAD).unsqueeze(0),
            batch_size=[1],
        )

    def ablate(self):
        self.lookup = torch.full_like(self.lookup, Tokens.UNK)
        self.day_of_week = torch.full_like(self.day_of_week, Tokens.UNK)
        self.day_of_year = torch.zeros_like(self.day_of_year)
        self.time_of_day = torch.zeros_like(self.time_of_day)


TensorField: TypeAlias = ContinuousField | DiscreteField | EntityField | TemporalField


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
