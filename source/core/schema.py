from typing import TypeAlias
from collections import namedtuple
import re

from beartype import beartype
from jaxtyping import Float, Int, Bool, jaxtyped
import torch
from torch import Tensor
from tensordict import TensorDict
from tensordict.prototype import tensorclass
import numpy as np
from param import Parameterized
import param

tokens = ["VAL_", "NAN_", "UNK_", "PAD_", "MASK_"]
SpecialTokens = namedtuple("SpecialTokens", tokens)
SPECIAL_TOKENS = SpecialTokens(*list(range(len(tokens))))

class Field:

    def __init__(self, **fieldinfo):
        
        assert len(fieldinfo), 'enter one field name and type'
        
        self.name: str
        self.type: str
        self.n_levels: None|int=None
        
        self.name, self.type = fieldinfo.popitem()
        
        assert self.type in [
            'continuous', 'discrete', 'entity', 'temporal'
        ]
        
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
        
        self.fields: list[Field] = []
        
        for name, dtype in fields.items():
            self.fields.append(Field(**{name: dtype}))
            
    def __len__(self) -> int:
        return len(self.fields)

class Hyperparameters(Parameterized):

    # architectural
    n_fields: int = param.Integer(bounds=(0, 128), default=None, allow_None=False)
    batch_size: int = param.Integer(192, bounds=(1, 2048))
    n_context: int = param.Integer(128, bounds=(1, 2048))
    d_field: int = param.Integer(64, bounds=(2, 256))
    n_heads_field_encoder: int = param.Integer(16, bounds=(1, 32))
    n_layers_field_encoder: int = param.Integer(2, bounds=(1, 32))
    n_heads_event_encoder: int = param.Integer(16, bounds=(1, 32))
    n_layers_event_encoder: int = param.Integer(8, bounds=(1, 32))
    precision: int = param.Integer(8, bounds=(2, 512))
    dropout: float = param.Magnitude(0.1)

    # general fitting
    dev_mode: bool = param.Boolean(False)
    learning_rate: float = param.Magnitude(0.0001)
    weight_decay: float = param.Magnitude(0.001)
    gradient_clip_val: float = param.Magnitude(0.05)
    max_epochs: int = param.Integer(64, bounds=(1, 256))
    
    # pretraining
    n_quantiles: int = param.Integer(16, bounds=(1, 512))
    p_mask_event: float = param.Magnitude(0.05)
    p_mask_field: float = param.Magnitude(0.05)
    pretrain_sample_rate: float = param.Magnitude(0.005)

    # finetuning
    n_targets: int = param.Integer(2, bounds=(2, 2048))
    freeze_epochs: int = param.Integer(1, bounds=(1, 128))
    interpolation_rate: float = param.Magnitude(0.125)
    head_shape_log_base: int = param.Integer(4, bounds=(1, 32))
    finetune_sample_rate: float = param.Magnitude(0.02)

    @property
    def values(self) -> dict[str, float|int|bool]:

        return self.param.values()


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
        
        PAD_ = getattr(SPECIAL_TOKENS, "PAD_")
        padding = (params.n_context - len(values), 0)

        array = np.array(values, dtype=int)
        lookup = torch.nn.functional.pad(torch.tensor(array), pad=padding, value=PAD_).unsqueeze(0)
        
        return cls(lookup=lookup, batch_size = [1])

    def mask(self, is_event_masked: Tensor, params: Hyperparameters) -> TargetField:

        N, L = (params.batch_size, params.n_context)
        mask_token = torch.full((N, L), getattr(SPECIAL_TOKENS, "MASK_"))

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
        PAD_ = getattr(SPECIAL_TOKENS, "PAD_")
        VAL_ = getattr(SPECIAL_TOKENS, "VAL_")
        
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

        N, L = (params.batch_size, params.n_context)
        mask_token = torch.full((N, L), getattr(SPECIAL_TOKENS, "MASK_"))
        
        # fine out what to mask
        is_field_masked = torch.rand(N, L).lt(params.p_mask_field)
        is_masked = is_field_masked | is_event_masked

        # creating discrete targets
        quantiles = self.content.mul(params.n_quantiles).floor().long().add(len(SPECIAL_TOKENS))
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
    def from_stream(cls, batch: tuple[TensorDict, TensorDict], batch_size: int):
        
        inputs, targets = batch
        
        return cls(inputs=inputs, targets=targets, batch_size=[batch_size])

@jaxtyped(typechecker=beartype)
@tensorclass
class FinetuningData:

    inputs: TensorDict[TensorField]
    targets: Tensor
    
    @classmethod
    def from_stream(cls, batch: tuple[TensorDict, Tensor], batch_size: int):
        
        inputs, targets = batch
        
        return cls(inputs=inputs, targets=targets, batch_size=[batch_size])

@jaxtyped(typechecker=beartype)
@tensorclass
class InferenceData:

    inputs: TensorDict[TensorField]
    
    # FIXME fix sequence id / event id

    # sequence_id: list[str]
    # event_id: list[str]
    
    @classmethod
    def from_stream(cls, batch: tuple[TensorDict, Tensor], batch_size: int):
        
        inputs, _ = batch

        # sequence_id = batch.pop('sequence_id')
        # event_id = batch.pop('sequence_id')
        
        return cls(inputs=inputs, batch_size=[batch_size])


SequenceData: TypeAlias = PretrainingData|FinetuningData|InferenceData
