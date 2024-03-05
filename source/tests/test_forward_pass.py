import pytest
import torch
from tdigest import TDigest
from tensordict import TensorDict

from source.core.iterops import mock
from source.core.schema import Field, Hyperparameters
from source.core.architecture import SequenceModule

@pytest.fixture
def discrete_only_inputs() -> tuple[Hyperparameters, TensorDict]:
    
    fields: list[Field] = [
        Field("is_online", "discrete", n_levels=2),
        Field("is_foreign", "discrete", n_levels=2),
    ]
    
    params = Hyperparameters(fields=fields)
    
    lifestream = mock(params)
    
    return params, lifestream

def test_discrete_only_forward_pass(discrete_only_inputs):
    
    params, lifestream = discrete_only_inputs
    
    module = SequenceModule(datapath='.', params=params, digests={})
    
    prediction, reconstruction = module.forward(lifestream)
    
    assert isinstance(prediction, torch.Tensor)
    assert isinstance(reconstruction, TensorDict)

@pytest.fixture
def continuous_only_inputs() -> tuple[Hyperparameters, TensorDict, dict[str, TDigest]]:
    
    fields: list[Field] = [
        Field("amount", "continuous"),
        Field("timediff", "continuous"),
    ]
    
    params = Hyperparameters(fields=fields)
    
    lifestream = mock(params)
    
    digests = dict(amount=TDigest(), timediff=TDigest())
    
    return params, lifestream, digests

def test_continuous_only_forward_pass(continuous_only_inputs):
    
    params, lifestream, digests = continuous_only_inputs
    
    module = SequenceModule(datapath='.', params=params, digests=digests)
    
    prediction, reconstruction = module.forward(lifestream)
    
    assert isinstance(prediction, torch.Tensor)
    assert isinstance(reconstruction, TensorDict)

@pytest.fixture
def mixed_dtype_inputs() -> tuple[Hyperparameters, TensorDict, dict[str, TDigest]]:
    
    fields: list[Field] = [
        Field("event_code", "discrete", n_levels=5),
        Field("is_online", "discrete", n_levels=2),
        Field("amount", "continuous"),
        Field("timediff", "continuous"),
    ]
    
    params = Hyperparameters(fields=fields)
    
    lifestream = mock(params)
    
    digests = dict(amount=TDigest(), timediff=TDigest())
    
    return params, lifestream, digests

def test_mixed_field_type_forward_pass(mixed_dtype_inputs):
    
    params, lifestream, digests = mixed_dtype_inputs
    
    module = SequenceModule(datapath='.', params=params, digests=digests)
    
    prediction, reconstruction = module.forward(lifestream)
    
    assert isinstance(prediction, torch.Tensor)
    assert isinstance(reconstruction, TensorDict)