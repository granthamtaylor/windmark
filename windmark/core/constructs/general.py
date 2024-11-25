# Copyright Grantham Taylor.

import re
import functools
from collections import namedtuple
from enum import IntEnum, Enum
from typing import Annotated
from dataclasses import dataclass

import pydantic
from pytdigest import TDigest


class Tokens(IntEnum):
    """
    Enum class representing different types of tokens.
    """

    # Value token.
    VAL = 0
    # Unknown token.
    UNK = 1
    # Padding token.
    PAD = 2
    # Mask token.
    MASK = 3
    # Prune token.
    PRUNE = 4


class FieldType(namedtuple("Field", ["name", "is_static"]), Enum):
    """
    Represents the type of a field in a data structure.

    Each field type has a name and a flag indicating whether it is static or dynamic.
    Static fields have a fixed type, while dynamic fields can have different types depending on the data.
    """

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
class FieldRequest:
    """
    Represents a field request object.
    """

    name: str
    fieldtype: str

    @classmethod
    def new(cls, name: str, fieldtype: FieldType | str) -> "FieldRequest":
        """
        Creates a new FieldRequest object.

        Args:
            name (str): The name of the field.
            fieldtype (FieldType | str): The type of the field.

        Returns:
            FieldRequest: The created FieldRequest object.

        Raises:
            KeyError: If the fieldtype is not a valid field type.
            AssertionError: If the field name is invalid.
        """
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
        """
        Returns the FieldType of the field.

        Returns:
            FieldType: The FieldType of the field.
        """
        return FieldType[self.fieldtype]

    @functools.cached_property
    def is_static(self) -> bool:
        """
        Returns True if the field is static, False otherwise.

        Returns:
            bool: True if the field is static, False otherwise.
        """
        return FieldType[self.fieldtype].is_static


@dataclass
class LevelSet:
    """
    Represents a set of levels.
    """

    name: str
    levels: list[str]

    def __len__(self) -> int:
        """
        Returns the number of levels in the set.

        Returns:
            int: The number of levels in the set.
        """
        return len(self.levels)

    @functools.cached_property
    def mapping(self) -> dict[str, int]:
        """
        Generates a mapping of levels to their corresponding indices.

        Returns:
            dict[str, int]: A dictionary mapping levels to their indices.
        """
        mapping = {level: index + len(Tokens) for index, level in enumerate(self.levels)}
        mapping[None] = int(Tokens.UNK)

        return mapping


@dataclass
class Centroid:
    """
    Represents a centroid with a name, an array of floats, and a validity flag.
    """

    name: str
    array: list[list[float, 2]]

    @classmethod
    def from_digest(cls, name: str, digest: TDigest) -> "Centroid":
        """
        Creates a centroid from a TDigest object.

        Args:
            name (str): The name of the centroid.
            digest (TDigest): The TDigest object containing the centroid data.

        Returns:
            Centroid: A centroid object created from the TDigest.

        """
        array = digest.get_centroids().tolist()
        return cls(name=name, array=array)


@pydantic.dataclasses.dataclass
class Hyperparameters:
    """Hyperparameters class for defining the model's configuration."""

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
    gradient_clip_val: Annotated[float, pydantic.Field(gt=0.0)]
    """Gradient clipping threshold"""
    max_pretrain_epochs: Annotated[int, pydantic.Field(gt=0, le=4096)]
    """Maximum number of epochs for pretraining"""
    max_finetune_epochs: Annotated[int, pydantic.Field(gt=0, le=4096)]
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
    predict_only_sequence_end: bool
    """Predict only last event in sequence"""

    @pydantic.model_validator(mode="after")
    def check_head_shape(self):
        """
        Checks if the head shape is valid.

        Raises:
            AssertionError: If the head shape is not valid.

        Returns:
            self: The current instance.
        """
        assert self.d_field % self.n_heads_field_encoder == 0, "d_field must be divisible by n_heads_field_encoder"
        assert self.d_field % self.n_heads_event_encoder == 0, "d_field must be divisible by n_heads_event_encoder"

        return self

    @pydantic.model_validator(mode="after")
    def check_finetuning_unfreeze(self):
        """
        Checks if the number of frozen epochs is less than the maximum finetune epochs.

        Returns:
            self: The current instance of the class.
        """
        assert self.n_epochs_frozen < self.max_finetune_epochs, "n_epochs_frozen must be less than max_finetune_epochs"

        return self

    @pydantic.model_validator(mode="after")
    def check_mask_rates(self):
        """
        Checks the masking rates and ensures they are above a minimum threshold.

        Returns:
            self: The current instance of the class.

        Raises:
            AssertionError: If the masking rates are below the minimum threshold.
        """
        rate = self.p_mask_field + self.p_mask_event + self.p_mask_static
        assert rate >= 0.01, "the masking rates are too low for any meaningful pretraining"

        return self
