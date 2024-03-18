from typing import TypeAlias
from dataclasses import dataclass
from collections import namedtuple
import re

from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
import torch
from torch import Tensor
from tensordict import TensorDict
from tensordict.prototype import tensorclass
import numpy as np
from dataclasses_json import dataclass_json
from param import Parameterized
import param

tokens = ["VAL_", "NAN_", "UNK_", "PAD_", "MASK_"]
SpecialTokens = namedtuple("SpecialTokens", tokens)
SPECIAL_TOKENS = SpecialTokens(*list(range(len(tokens))))

@dataclass_json
@dataclass
class Field:
    name: str
    dtype: str
    n_levels: None | int = None

    def __post_init__(self):
        assert re.match("^[a-z0-9_]*$", self.name), "name must only contain lowercase letters, numbers, and underscore"

        if isinstance(self.n_levels, float):
            self.n_levels = int(self.n_levels)

        assert self.dtype in ["discrete", "continuous", "entity"]

        match self.dtype:
            case "discrete":
                assert isinstance(self.n_levels, int), "n_levels must be an int for discrete fields"
                assert self.n_levels > 0, "n_levels must be greater than 0"

            case "continuous":
                assert self.n_levels is None, "n_levels must be none for continuous fields"

            case "entity":
                assert self.n_levels is None, "n_levels must be none for entity fields"    

class TrainingParameters(Parameterized):

    max_epochs: int = param.Integer(4, bounds=(1, 1024))
    sample_rate: float = param.Magnitude(0.001)
    check_val_every_n_epoch: int = param.Integer(1, bounds=(1, 32))
    gradient_clip_val: float = param.Magnitude(0.05)

class Hyperparameters(Parameterized):

    dev_mode: bool = param.Boolean(False)
    fields: list[Field] = param.List(item_type=Field)
    learning_rate: float = param.Magnitude(0.0001)

    # architecture hyperparameters
    batch_size: int = param.Integer(128, bounds=(1, 2048))
    n_context: int = param.Integer(128, bounds=(1, 2048))
    d_field: int = param.Integer(64, bounds=(2, 256))
    n_heads_field_encoder: int = param.Integer(16, bounds=(1, 32))
    n_layers_field_encoder: int = param.Integer(2, bounds=(1, 32))
    n_heads_event_encoder: int = param.Integer(16, bounds=(1, 32))
    n_layers_event_encoder: int = param.Integer(8, bounds=(1, 32))
    precision: int = param.Integer(8, bounds=(2, 512))
    dropout: float = param.Magnitude(0.1)
    
    # pretraining
    n_quantiles: int = param.Integer(16, bounds=(1, 512))
    p_mask_event: float = param.Magnitude(0.1)
    p_mask_field: float = param.Magnitude(0.1)
    pretrain: TrainingParameters = param.Parameter(TrainingParameters(
        max_epochs=2,
        sample_rate=0.01,
        check_val_every_n_epoch=8,
        gradient_clip_val=0.05,
    ))

    # finetuning
    n_targets: int = param.Integer(2, bounds=(2, 2048))
    freeze_epochs: int = param.Integer(1, bounds=(1, 128))
    interpolation_rate: float = param.Magnitude(0.075)
    head_shape_log_base: int = param.Integer(4, bounds=(1, 32))
    finetune: TrainingParameters = param.Parameter(TrainingParameters(
        max_epochs=72,
        sample_rate=0.1,
        check_val_every_n_epoch=1,
        gradient_clip_val=0.05,
    ))


    @property
    def n_fields(self) -> int:

        return len(self.fields)

    def values(self):

        output = self.param.values()
        
        del output['name']

        output['pretrain'] = self.finetune.param.values()
        del output['pretrain']['name']

        output['finetune'] = self.finetune.param.values()
        del output['finetune']['name']
        
        return output

@tensorclass
class DiscreteField:
    lookup: Int[Tensor, "N L"]

    @classmethod
    def collate(cls, values: list[int], params: Hyperparameters):
        
        PAD_ = getattr(SPECIAL_TOKENS, "PAD_")
        padding = (params.n_context - len(values), 0)

        array = np.array(values, dtype=int)
        lookup = torch.nn.functional.pad(torch.tensor(array), pad=padding, value=PAD_).unsqueeze(0)
        
        return cls(lookup=lookup, batch_size = [1])

    def mask(self, is_event_masked: Tensor, params: Hyperparameters):

        N, L = (params.batch_size, params.n_context)
        mask_token = torch.full((N, L), getattr(SPECIAL_TOKENS, "MASK_"))
        
        is_field_masked = torch.rand(N, L).lt(params.p_mask_field)
        
        for mask in [is_field_masked, is_event_masked]:
            self.lookup.masked_scatter_(mask, mask_token)
            
    def target(self, params: Hyperparameters) -> Tensor:
        return self.lookup

@tensorclass
class EntityField:
    lookup: Int[Tensor, "N L"]

    collate = DiscreteField.collate
    mask = DiscreteField.mask

@tensorclass
class ContinuousField:
    lookup: Int[Tensor, "N L"]
    content: Float[Tensor, "N L"]

    @classmethod
    def collate(cls, values, params: Hyperparameters):
        
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
    
    def mask(self, is_event_masked: Tensor, params: Hyperparameters):

        N, L = (params.batch_size, params.n_context)
        mask_token = torch.full((N, L), getattr(SPECIAL_TOKENS, "MASK_"))
        
        is_field_masked = torch.rand(N, L).lt(params.p_mask_field)
        
        for mask in [is_field_masked, is_event_masked]:

            self.lookup.masked_scatter_(mask, mask_token)
            self.content *= ~mask
        
    def target(self, params: Hyperparameters) -> Tensor:
        
        quantiles = self.content.mul(params.n_quantiles).floor().long().add(len(SPECIAL_TOKENS))
        is_not_valued = self.lookup != 0
        return quantiles.masked_scatter(is_not_valued, quantiles)

TensorField: TypeAlias = ContinuousField|DiscreteField|EntityField

@tensorclass
class PretrainingData:

    inputs: TensorDict[TensorField]
    targets: TensorDict[Tensor]
    
    @classmethod
    def from_stream(cls, batch: tuple[TensorDict, TensorDict], batch_size: int):
        
        inputs, targets = batch
        
        return cls(inputs=inputs, targets=targets, batch_size=[batch_size])

@tensorclass
class FinetuningData:

    inputs: TensorDict[TensorField]
    targets: Tensor
    
    @classmethod
    def from_stream(cls, batch: tuple[TensorDict, Tensor], batch_size: int):
        
        inputs, targets = batch
        
        return cls(inputs=inputs, targets=targets, batch_size=[batch_size])
    

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
