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
import pydantic
from rich.console import Console
from rich.table import Table

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
    n_fields: int = pydantic.Field(..., gt=0, lt=129)
    batch_size: int = pydantic.Field(72, gt=0, lt=2049)
    n_context: int = pydantic.Field(128, gt=0, lt=2049)
    d_field: int = pydantic.Field(64, gt=1, lt=257)
    precision: int = pydantic.Field(8, gt=1, lt=513)
    n_heads_field_encoder: int = pydantic.Field(16, gt=0, lt=33)
    n_layers_field_encoder: int = pydantic.Field(2, gt=0, lt=33)
    n_heads_event_encoder: int = pydantic.Field(16, gt=0, lt=33)
    n_layers_event_encoder: int = pydantic.Field(8, gt=0, lt=33)
    dropout: float = pydantic.Field(0.1, gt=0.0, lt=1.0)

    # general fitting
    learning_rate: float = pydantic.Field(0.0001, gt=0.0, lt=1.0)
    weight_decay: float = pydantic.Field(0.001, gt=0.0, lt=1.0)
    gradient_clip_val: float = pydantic.Field(0.05, gt=0.0, lt=1.0)
    max_epochs: int = pydantic.Field(1, gt=0, lt=257)
    
    # pretraining
    n_quantiles: int = pydantic.Field(16, gt=0, lt=513)
    sigma: float = pydantic.Field(1.0, gt=0.0, lt=33.0)
    p_mask_event: float = pydantic.Field(0.05, gt=0.0, lt=1.0)
    p_mask_field: float = pydantic.Field(0.05, gt=0.0, lt=1.0)
    pretrain_sample_rate: float = pydantic.Field(0.001, gt=0.0, lt=1.0)

    # finetuning
    n_targets: int = pydantic.Field(2, gt=1, lt=2049)
    freeze_epochs: int = pydantic.Field(1, gt=0, lt=129)
    interpolation_rate: float = pydantic.Field(0.125, gt=0.0, lt=1.0)
    head_shape_log_base: int = pydantic.Field(4, gt=0, lt=33)
    finetune_sample_rate: float = pydantic.Field(0.1, gt=0.0, lt=1.0)

    @property
    def values(self) -> dict[str, float|int|bool]:

        params = self.param.values()
        del params['name']
        
        return params
        

    def show(self):
        
        table = Table(title="Hyperparameters")

        table.add_column("Hyperparameter", justify="right", style="cyan", no_wrap=True)
        
        table.add_column('Value', style="magenta")
            
        def format_percent(values: list[float]) -> list[str]:
            return list(map(lambda x: f"{x:.4%}", values))
        
        def format_numbers(values: list[float]) -> list[str]:
            return list(map(lambda x: f"{x:.4}", values))
        
        def format_integers(values: list[float]) -> list[str]:
            return list(map(lambda x: f"{x:,}", values))

        # table.add_row("Label Counts", *format_integers(self.counts))
        # table.add_row("Population Distribution", *format_percent(self.values))
        # table.add_row("Observation Distribution", *format_percent(self.interpolation))
        # table.add_row("Marginal Sample Rate", *format_percent(self.thresholds))
        # table.add_row("Loss Weights", *format_numbers(self.weights))

        
        for param, value in self.values.items():
            table.add_row(param, str(value))

        console = Console()
        console.print(table)

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

        N, L = (1, params.n_context)
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

        N, L = (1, params.n_context)
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
