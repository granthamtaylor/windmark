import re
import functools
from collections import namedtuple
from enum import IntEnum, Enum
from typing import Annotated
from dataclasses import dataclass

import pydantic
from pytdigest import TDigest
from mashumaro.mixins.json import DataClassJSONMixin


class Tokens(IntEnum):
    """Special token representations"""

    VAL = 0
    UNK = 1
    PAD = 2
    MASK = 3
    PRUNE = 4


class FieldType(namedtuple("Field", ["name", "is_static"]), Enum):
    # dynamic
    Numbers = ("Numbers", False)
    Categories = ("Categories", False)
    Timestamps = ("Timestamps", False)
    Quantiles = ("Quantiles", False)
    Entities = ("Entities", False)

    # static
    Number = ("Number", True)
    Category = ("Category", True)
    Timestamp = ("Timestamp", True)
    Quantile = ("Quantile", True)

    def __str__(self) -> str:
        return self.name


@dataclass
class FieldRequest(DataClassJSONMixin):
    name: str
    fieldtype: str

    @classmethod
    def new(cls, name: str, fieldtype: FieldType | str) -> "FieldRequest":
        if isinstance(fieldtype, str):
            # check if valid field type
            if fieldtype.capitalize() in FieldType._member_names_:
                fieldtype: FieldType = FieldType[fieldtype.capitalize()]
            else:
                raise KeyError

        assert re.match(r"^[a-z][a-z0-9_]*$", name), f"invalid field name {name}"

        return cls(name=name, fieldtype=str(fieldtype))

    @functools.cached_property
    def type(self) -> FieldType:
        return FieldType[self.fieldtype]

    @functools.cached_property
    def is_static(self) -> bool:
        return FieldType[self.fieldtype].is_static


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
        mapping[None] = int(Tokens.UNK)

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
    batch_size: Annotated[int, pydantic.Field(gt=0, le=2048)]
    """Batch size for training (how many observations per step)"""
    n_context: Annotated[int, pydantic.Field(gt=0, le=2048)]
    """Context size (how many events per observation)"""
    d_field: Annotated[int, pydantic.Field(gt=1, le=256)]
    """Hidden dimension per field"""
    n_heads_field_encoder: Annotated[int, pydantic.Field(gt=0, le=32)]
    """Number of heads in field encoder"""
    n_layers_field_encoder: Annotated[int, pydantic.Field(gt=0, le=32)]
    """Number of layers in field encoder"""
    n_heads_event_encoder: Annotated[int, pydantic.Field(gt=0, le=32)]
    """Number of heads in event encoder"""
    n_layers_event_encoder: Annotated[int, pydantic.Field(gt=0, le=32)]
    """Number of layers in event encoder"""
    dropout: Annotated[float, pydantic.Field(ge=0.0, lt=1.0)]
    """Dropout rate"""
    n_bands: Annotated[int, pydantic.Field(gt=1, le=16)]
    """Precision of fourier feature encoders"""
    head_shape_log_base: Annotated[int, pydantic.Field(gt=1, le=8)]
    """How quickly to converge sequence representation"""
    n_quantiles: Annotated[int, pydantic.Field(gt=1, le=512)]
    """Number of quantiles for continuous and temporal field"""

    # training
    n_pretrain_steps: Annotated[int, pydantic.Field(gt=0)]
    """Number of steps to take per epoch during pretraining"""
    n_finetune_steps: Annotated[int, pydantic.Field(gt=0)]
    """Number of steps to take per epoch during finetuning"""
    swa_lr: Annotated[float, pydantic.Field(ge=0.0, lt=1.0)]
    """Stochastic Weight Averaging"""
    gradient_clip_val: Annotated[float, pydantic.Field(gt=0.0)]
    """Gradient clipping threshold"""
    max_pretrain_epochs: Annotated[int, pydantic.Field(gt=0, le=1028)]
    """Maximum number of epochs for pretraining"""
    max_finetune_epochs: Annotated[int, pydantic.Field(gt=0, le=1028)]
    """Maximum number of epochs for finetuning"""
    quantile_smoothing: Annotated[float, pydantic.Field(gt=0.0, lt=33.0)]
    """Smoothing factor of continuous fields' quantile labels"""
    p_mask_event: Annotated[float, pydantic.Field(ge=0.0, lt=1.0)]
    """Probability of masking any event"""
    p_mask_field: Annotated[float, pydantic.Field(ge=0.0, lt=1.0)]
    """Probability of masking any dynamic field"""
    p_mask_static: Annotated[float, pydantic.Field(ge=0.0, lt=1.0)]
    """Probability of masking any static field"""
    n_epochs_frozen: Annotated[int, pydantic.Field(gt=0, le=128)]
    """Number of epochs to freeze encoder while finetuning"""
    interpolation_rate: Annotated[float, pydantic.Field(ge=0.0, le=1.0)]
    """Interpolation rate of imbalanced classification labels"""
    learning_rate: Annotated[float, pydantic.Field(gt=0.0, lt=1.0)]
    """Learning Rate during Pretraining"""
    learning_rate_dampener: Annotated[float, pydantic.Field(gt=0.0, lt=1.0)]
    """Learning Rate Modifier during Finetuning"""
    patience: Annotated[int, pydantic.Field(ge=1, le=256)]
    """Number of Epochs Patience for Early Stopping"""
    jitter: Annotated[float, pydantic.Field(ge=0.0, lt=1.0)]
    """Amount of jitter to apply to continuous values"""
    n_workers: Annotated[int, pydantic.Field(ge=1, le=256)]
    """Number of parallelized data pipes"""
    predict_only_sequence_end: bool
    """Predict only last event in sequence"""

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
        rate = self.p_mask_field + self.p_mask_event + self.p_mask_static
        assert rate >= 0.01, "the masking rates are too low for any meaningful pretraining"

        return self
