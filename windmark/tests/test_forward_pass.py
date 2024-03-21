import pytest
import torch
from pytdigest import TDigest
from tensordict import TensorDict
import numpy as np

from windmark.core.utils import mock
from windmark.core.schema import Field, Hyperparameters
from windmark.core.architecture import SequenceModule

@pytest.fixture
def discrete_only_inputs() -> tuple[Hyperparameters, TensorDict]:
    
    fields: list[Field] = [
        Field(is_online="discrete"),
        Field(is_foreign="discrete"),
    ]
    
    for field in fields:
        field.levels = 2
    
    params = Hyperparameters(n_fields = len(fields))
    
    lifestream = mock(params=params, fields=fields)
    
    return params, lifestream

def test_discrete_only_forward_pass(discrete_only_inputs):
    
    params, lifestream = discrete_only_inputs
    
    module = SequenceModule(datapath='.', params=params, digests={})
    
    prediction, reconstruction = module.forward(lifestream)
    
    assert isinstance(prediction, torch.Tensor)
    assert isinstance(reconstruction, TensorDict)

@pytest.fixture
def continuous_only_inputs() -> tuple[Hyperparameters, TensorDict, dict[str, np.ndarray]]:
    
    fields: list[Field] = [
        Field(amount="continuous"),
        Field(timediff="continuous"),
    ]
    
    params = Hyperparameters(n_fields = len(fields))
    
    lifestream = mock(params=params, fields=fields)
    
    centroids = dict(amount=TDigest().get_centroids(), timediff=TDigest().get_centroids())
    
    return params, lifestream, centroids

def test_continuous_only_forward_pass(continuous_only_inputs):
    
    params, lifestream, centroids = continuous_only_inputs
    
    module = SequenceModule(datapath='.', params=params, centroids=centroids)
    
    prediction, reconstruction = module.forward(lifestream)
    
    assert isinstance(prediction, torch.Tensor)
    assert isinstance(reconstruction, TensorDict)

@pytest.fixture
def mixed_type_inputs() -> tuple[Hyperparameters, TensorDict, dict[str, np.ndarray]]:
    
    fields: list[Field] = [
        Field(event_code="discrete"),
        Field(is_online="discrete"),
        Field(amount="continuous"),
        Field(timediff="continuous"),
    ]
    
    for field in fields:
        if field.type == 'discrete':
            field.levels = 5
    
    params = Hyperparameters(n_fields = len(fields))
    
    lifestream = mock(params=params, fields=fields)
    
    centroids = dict(amount=TDigest().get_centroids(), timediff=TDigest().get_centroids())
    
    return params, lifestream, centroids

def test_mixed_field_type_forward_pass(mixed_type_inputs):
    
    params, lifestream, centroids = mixed_type_inputs
    
    module = SequenceModule(datapath='.', params=params, centroids=centroids)
    
    prediction, reconstruction = module.forward(lifestream)
    
    assert isinstance(prediction, torch.Tensor)
    assert isinstance(reconstruction, TensorDict)