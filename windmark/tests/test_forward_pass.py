import torch
from pytdigest import TDigest
from tensordict import TensorDict

from windmark.core.architecture import SequenceModule
from windmark.core.operators import mock
from windmark.core.structs import Field, Hyperparameters
from windmark.core.managers import ClassificationManager


def test_discrete_only_forward_pass():
    fields: list[Field] = [
        Field(is_online="discrete"),
        Field(is_foreign="discrete"),
    ]

    for field in fields:
        field.levels = 2

    params = Hyperparameters(n_fields=len(fields))

    lifestream = mock(params=params, fields=fields)

    balancer = ClassificationManager(labels=["is_fraud", "not_fraud"], counts=[31, 312], kappa=0.5)

    module = SequenceModule(datapath=".", fields=fields, params=params, centroids={}, balancer=balancer)

    prediction, reconstruction = module.forward(lifestream)

    assert isinstance(prediction, torch.Tensor)
    assert isinstance(reconstruction, TensorDict)


def test_continuous_only_forward_pass():
    fields: list[Field] = [
        Field(amount="continuous"),
        Field(timediff="continuous"),
    ]

    params = Hyperparameters(n_fields=len(fields))

    lifestream = mock(params=params, fields=fields)

    centroids = dict(amount=TDigest().get_centroids(), timediff=TDigest().get_centroids())

    balancer = ClassificationManager(labels=["is_fraud", "not_fraud"], counts=[31, 31], kappa=0.5)

    module = SequenceModule(datapath=".", fields=fields, params=params, centroids=centroids, balancer=balancer)

    prediction, reconstruction = module.forward(lifestream)

    assert isinstance(prediction, torch.Tensor)
    assert isinstance(reconstruction, TensorDict)


def test_mixed_field_type_forward_pass():
    fields: list[Field] = [
        Field(event_code="discrete"),
        Field(is_online="discrete"),
        Field(amount="continuous"),
        Field(timediff="continuous"),
    ]

    for field in fields:
        if field.type == "discrete":
            field.levels = 5

    params = Hyperparameters(n_fields=len(fields))

    lifestream = mock(params=params, fields=fields)

    centroids = dict(amount=TDigest().get_centroids(), timediff=TDigest().get_centroids())

    balancer = ClassificationManager(labels=["is_fraud", "not_fraud"], counts=[31, 31], kappa=0.5)

    module = SequenceModule(datapath=".", fields=fields, params=params, centroids=centroids, balancer=balancer)

    prediction, reconstruction = module.forward(lifestream)

    assert isinstance(prediction, torch.Tensor)
    assert isinstance(reconstruction, TensorDict)
