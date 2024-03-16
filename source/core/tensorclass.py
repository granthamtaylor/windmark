from typing import TypeAlias

from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
import torch
from torch import Tensor
from tensordict import TensorDict
from tensordict.prototype import tensorclass
import numpy as np

from source.core.schema import SPECIAL_TOKENS, Hyperparameters
    

@tensorclass
class DiscreteField:
    lookup: Int[Tensor, "N L"]

    @classmethod
    def collate(cls, values: list[int], params: Hyperparameters):
        
        PAD_ = getattr(SPECIAL_TOKENS, "PAD_")
        padding = (params.n_context - len(values), 0)

        array = np.array(values, dtype=int)
        lookup = torch.nn.functional.pad(torch.tensor(array), pad=padding, value=PAD_).unsqueeze(0)
        
        return cls(lookup=lookup, batch_size = [params.batch_size])

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
            batch_size = [params.batch_size],
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

    # sequence_id: list[str]
    # event_id: list[str]
    
    @classmethod
    def from_stream(cls, batch: tuple[TensorDict, Tensor], batch_size: int):
        
        inputs, _ = batch

        # sequence_id = batch.pop('sequence_id')
        # event_id = batch.pop('sequence_id')
        
        return cls(inputs=inputs, batch_size=[batch_size])


SequenceData: TypeAlias = PretrainingData|FinetuningData|InferenceData