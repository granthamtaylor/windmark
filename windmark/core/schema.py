from typing import TypeAlias
from enum import IntEnum
import re

from beartype import beartype
from jaxtyping import Float, Int, Bool, jaxtyped
import torch
from torch import Tensor
from tensordict import TensorDict
from tensordict.prototype import tensorclass
import numpy as np
import pydantic

class SpecialTokens(IntEnum):

    VAL = 0
    NAN = 1
    UNK = 2
    PAD = 3
    MASK = 4

class Field:

    def __init__(self, **fieldinfo):
        
        assert len(fieldinfo), 'enter one field name and type'
        
        self.name: str
        self.type: str
        self.n_levels: None|int=None
        
        self.name, self.type = fieldinfo.popitem()
        
        assert self.type in ['continuous', 'discrete', 'entity', 'temporal'], \
            'field type must be "continuous", "discrete", "entity", or "temporal"'
        
        assert re.match(r'^[a-z][a-z0-9_]*$', self.name), f'invalid field name {self.name}'
    
    @property
    def levels(self):

        return self.n_levels

    @levels.setter
    def levels(self, value: int):
        
        assert isinstance(value, int), 'value must be of type int'
        assert self.type == 'discrete', 'only discrete fields can have this attribute'
        
        self.n_levels = value
    
    @property
    def is_valid(self):

        if self.type == 'discrete':
            return isinstance(self.n_levels, int)
        else:
            return True

class Schema:
    
    def __init__(self, **fields):
        
        assert len(fields) > 1, 'must pass in at least two fields'
        
        self.fields: list[Field] = []
        
        for name, dtype in fields.items():
            self.fields.append(Field(**{name: dtype}))
            
    def __len__(self) -> int:

        return len(self.fields)

class Hyperparameters(pydantic.BaseModel):

    # architectural
    n_fields: int = pydantic.Field(..., gt=1, lt=129, description="Number of unique fields")
    batch_size: int = pydantic.Field(72, gt=0, lt=2049, description="Batch size for training (how many observations per step")
    n_context: int = pydantic.Field(128, gt=0, lt=2049, description="Context size (how many events per observation)")
    d_field: int = pydantic.Field(64, gt=1, lt=257, description="Hidden dimension per field")
    precision: int = pydantic.Field(8, gt=1, lt=513, description="Precision of fourier feature encoders")
    n_heads_field_encoder: int = pydantic.Field(16, gt=0, lt=33, description="Number of heads in field encoder", )
    n_layers_field_encoder: int = pydantic.Field(2, gt=0, lt=33, description="Number of layers in field encoder")
    n_heads_event_encoder: int = pydantic.Field(16, gt=0, lt=33, description="Number of heads in event encoder", )
    n_layers_event_encoder: int = pydantic.Field(8, gt=0, lt=33, description="Number of layers in event encoder")
    dropout: float = pydantic.Field(0.1, gt=0.0, lt=1.0, description="Dropout rate")

    # general fitting
    learning_rate: float = pydantic.Field(0.0001, gt=0.0, lt=1.0, description="Learning rate (will be overwritten)")
    weight_decay: float = pydantic.Field(0.001, gt=0.0, lt=1.0, description="Optimizer weight decay")
    gradient_clip_val: float = pydantic.Field(0.05, gt=0.0, lt=1.0, description="Gradient clipping threshold")
    max_epochs: int = pydantic.Field(1, gt=0, lt=257, description="Maximum number of epochs for pretraining and finetuning")
    
    # pretraining
    n_quantiles: int = pydantic.Field(16, gt=0, lt=513, description="Number of quantiles for continuous and temporal field")
    sigma: float = pydantic.Field(1.0, gt=0.0, lt=33.0, description="Smoothing factor of continuous fields' self-supervised")
    p_mask_event: float = pydantic.Field(0.05, gt=0.0, lt=1.0, description="Probability of masking any event")
    p_mask_field: float = pydantic.Field(0.05, gt=0.0, lt=1.0, description="Probability of masking any field")
    pretrain_sample_rate: float = pydantic.Field(0.001, gt=0.0, lt=1.0, description="Proportion of events to sample from during pretraining")

    # finetuning
    n_targets: int = pydantic.Field(2, gt=1, lt=2049, description="Number of supervised classification targets")
    freeze_epochs: int = pydantic.Field(1, gt=0, lt=129, description="Number of epochs to freeze encoder while finetuning")
    interpolation_rate: float = pydantic.Field(0.125, gt=0.0, lt=1.0, description="Interpolation rate of imbalanced classification labels")
    head_shape_log_base: int = pydantic.Field(4, gt=0, lt=33, description="How quickly to converge sequence representation")
    finetune_sample_rate: float = pydantic.Field(0.1, gt=0.0, lt=1.0, description="Proportion of events to sample from during finetuning")

@jaxtyped(typechecker=beartype)
@tensorclass
class TargetField:
    
    lookup: Int[Tensor, "N L"]
    is_masked: Bool[Tensor, "N L"]
    

@jaxtyped(typechecker=beartype)
@tensorclass
class DiscreteField:

    lookup: Int[Tensor, "N L"]

    @classmethod
    def collate(cls, values: list[int], params: Hyperparameters) -> "DiscreteField":
        
        PAD_ = int(SpecialTokens.PAD)
        padding = (params.n_context - len(values), 0)

        array = np.array(values, dtype=int)
        lookup = torch.nn.functional.pad(torch.tensor(array), pad=padding, value=PAD_).unsqueeze(0)
        
        return cls(lookup=lookup, batch_size = [1])

    def mask(self, is_event_masked: Tensor, params: Hyperparameters) -> TargetField:

        N, L = (1, params.n_context)
        mask_token = torch.full((N, L), int(SpecialTokens.MASK))

        is_field_masked = torch.rand(N, L).lt(params.p_mask_field)
        is_masked = is_field_masked.logical_or(is_event_masked)
        
        targets = self.lookup.clone()

        self.lookup = self.lookup.masked_scatter(is_masked, mask_token)

        return TargetField(
            lookup=targets,
            is_masked=is_masked,
            batch_size=self.batch_size,
        )


@jaxtyped(typechecker=beartype)
@tensorclass
class EntityField:

    lookup: Int[Tensor, "N L"]

    collate = DiscreteField.collate
    mask = DiscreteField.mask

@jaxtyped(typechecker=beartype)
@tensorclass
class ContinuousField:
    lookup: Int[Tensor, "N L"]
    content: Float[Tensor, "N L"]

    @classmethod
    def collate(cls, values, params: Hyperparameters) -> "ContinuousField":
        
        padding = (params.n_context - len(values), 0)
        PAD_ = int(SpecialTokens.PAD)
        VAL_ = int(SpecialTokens.VAL)
        
        values = np.nan_to_num(np.array(values, dtype=float))
        lookup = np.where(np.isnan(values), PAD_, VAL_)
        
        # this effectively creates `1-(1/inf)` to prevent an index error
        # somewhere in the dataset this exists a CDF of `1.0`, which will not be "floored" correctly
        dampener = 1 - torch.finfo(torch.half).tiny

        return cls(
            content = torch.nn.functional.pad(torch.tensor(values), pad=padding, value=0.0).float().unsqueeze(0).mul(dampener),
            lookup = torch.nn.functional.pad(torch.tensor(lookup), pad=padding, value=PAD_).unsqueeze(0),
            batch_size = [1],
        )
    
    def mask(self, is_event_masked: Tensor, params: Hyperparameters) -> TargetField:

        N, L = (1, params.n_context)
        mask_token = torch.full((N, L), int(SpecialTokens.MASK))
        
        # fine out what to mask
        is_field_masked = torch.rand(N, L).lt(params.p_mask_field)
        is_masked = is_field_masked | is_event_masked

        # creating discrete targets
        quantiles = self.content.mul(params.n_quantiles).floor().long().add(len(SpecialTokens))
        is_not_valued = (self.lookup != 0)
        targets = quantiles.masked_scatter(is_not_valued, quantiles)

        # mask original values
        self.lookup = self.lookup.masked_scatter(is_masked, mask_token)
        self.content *= ~is_masked
        
        # return SSL target
        return TargetField(
            lookup=targets,
            is_masked=is_masked,
            batch_size=self.batch_size
        )

@jaxtyped(typechecker=beartype)
@tensorclass
class TemporalField:

    lookup: Int[Tensor, "N L"]
    content: Float[Tensor, "N L"]

    collate = ContinuousField.collate
    mask = ContinuousField.mask

TensorField: TypeAlias = ContinuousField|DiscreteField|EntityField|TemporalField

@jaxtyped(typechecker=beartype)
@tensorclass
class PretrainingData:

    inputs: TensorDict[TensorField]
    targets: TensorDict[Tensor]
    
    @classmethod
    def from_stream(cls, batch: tuple[TensorDict, TensorDict, tuple[str, str]]):
        
        inputs, targets, _ = batch
        
        return cls(inputs=inputs, targets=targets, batch_size=[1])

@jaxtyped(typechecker=beartype)
@tensorclass
class FinetuningData:

    inputs: TensorDict[TensorField]
    targets: Tensor
    
    @classmethod
    def from_stream(cls, batch: tuple[TensorDict, Tensor, tuple[str, str]]):
        
        inputs, targets, _ = batch

        targets = targets.unsqueeze(0)
        
        return cls(inputs=inputs, targets=targets, batch_size=[1])

@jaxtyped(typechecker=beartype)
@tensorclass
class InferenceData:

    inputs: TensorDict[TensorField]
    meta: tuple[str, str]
    
    @classmethod
    def from_stream(cls, batch: tuple[TensorDict, Tensor, tuple[str, str]]):
        
        inputs, _, meta = batch
        
        return cls(inputs=inputs, meta=meta, batch_size=[1])


SequenceData: TypeAlias = PretrainingData|FinetuningData|InferenceData
