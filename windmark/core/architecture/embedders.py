import math
from typing import TypeAlias

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from windmark.core.constructs.general import Hyperparameters, Tokens, FieldRequest
from windmark.core.constructs.tensorfields import DiscreteField, ContinuousField, EntityField, TemporalField
from windmark.core.managers import SystemManager


class DiscreteFieldEmbedder(torch.nn.Module):
    def __init__(self, params: Hyperparameters, manager: SystemManager, field: FieldRequest):
        """
        Initialize discrete field embedder.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
            manager (SystemManager): The pipeline system manager.
            field (Field): The field to be embedded
        """
        super().__init__()

        self.field: FieldRequest = field
        self.embeddings = torch.nn.Embedding(manager.levelsets.get_size(field) + len(Tokens), params.d_field)

    def forward(self, inputs: DiscreteField) -> torch.Tensor:
        return self.embeddings(inputs.lookup)


class EntityFieldEmbedder(torch.nn.Module):
    def __init__(self, params: Hyperparameters, manager: SystemManager, field: FieldRequest):
        super().__init__()
        """
        Initialize entity field embedder.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
            manager (SystemManager): The pipeline system manager.
            field (Field): The field to be embedded
        """

        self.field: FieldRequest = field
        self.embeddings = torch.nn.Embedding(params.n_context + len(Tokens), params.d_field)

    def forward(self, inputs: EntityField) -> torch.Tensor:
        return self.embeddings(inputs.lookup)


class ContinuousFieldEmbedder(torch.nn.Module):
    """
    ContinuousFieldEmbedder is a PyTorch module that encodes features using Fourier features.

    Attributes:
        linear (torch.nn.Linear): A linear layer for transforming the input.
        positional (torch.nn.Embedding): An embedding layer for positional encoding.
        weights (Tensor): The weights for the Fourier features.
    """

    def __init__(self, params: Hyperparameters, manager: SystemManager, field: FieldRequest):
        """
        Initialize continuous field embedder.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
            manager (SystemManager): The pipeline system manager.
            field (Field): The field to be embedded
        """

        super().__init__()

        self.field: FieldRequest = field

        offset = 3

        weights = torch.logspace(start=-params.n_bands, end=offset, steps=params.n_bands + offset + 1, base=2)

        self.linear = torch.nn.Linear(2 * len(weights), params.d_field)
        self.register_buffer("weights", weights.mul(math.pi).unsqueeze(dim=0))

    @jaxtyped(typechecker=beartype)
    def forward(self, inputs: ContinuousField) -> Float[torch.Tensor, "N L F"]:
        """
        Performs the forward pass of the FourierFeatureEncoder.

        Args:
            inputs (Float[Tensor, "N L"]): The input tensor.

        Returns:
            Float[Tensor, "N L F"]: The Fourier features of the input.
        """

        values = inputs.content
        indicators = inputs.lookup

        assert values.shape == indicators.shape, "values and indicators must always have the same shape"

        assert torch.all(
            values.mul(indicators).eq(0.0), dim=None
        ), "values should be imputed if not null, padded, or masked"

        assert torch.all(values.lt(1.0), dim=None), "values should be less than 1.0"

        assert torch.all(values.ge(0.0), dim=None), "values should be greater than or equal to 0.0"

        N, L = values.shape

        # weight inputs with buffers of precision bands
        weighted = values.sub(indicators).view(N * L).unsqueeze(dim=1).mul(self.weights)

        # apply sine and cosine functions to weighted inputs
        fourier = torch.sin(weighted), torch.cos(weighted)

        # project sinusoidal representations with MLP
        projections = self.linear(torch.cat(fourier, dim=1)).view(N, L, -1)

        return projections


class TemporalFieldEmbedder(torch.nn.Module):
    def __init__(self, params: Hyperparameters, manager: SystemManager, field: FieldRequest):
        """
        Initialize continuous field embedder.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
            manager (SystemManager): The pipeline system manager.
            field (Field): The field to be embedded
        """

        super().__init__()

        self.field: FieldRequest = field

        offset = 0

        weights = torch.logspace(start=-params.n_bands, end=offset, steps=params.n_bands + offset + 1, base=2)

        self.linear = torch.nn.Linear(2 * len(weights), params.d_field)
        self.register_buffer("weights", weights.mul(math.pi).unsqueeze(dim=0))

        self.embeddings = torch.nn.ModuleDict(
            dict(
                week_of_year=torch.nn.Embedding(53 + len(Tokens), params.d_field),
                day_of_week=torch.nn.Embedding(7 + len(Tokens), params.d_field),
            )
        )

    @jaxtyped(typechecker=beartype)
    def forward(self, inputs: TemporalField) -> Float[torch.Tensor, "N L F"]:
        """
        Performs the forward pass of the FourierFeatureEncoder.

        Args:
            inputs (Float[Tensor, "N L"]): The input tensor.

        Returns:
            Float[Tensor, "N L F"]: The Fourier features of the input.
        """

        assert inputs.time_of_day.shape == inputs.lookup.shape, "values and indicators must always have the same shape"

        assert torch.all(
            inputs.time_of_day.mul(inputs.lookup).eq(0.0), dim=None
        ), "values should be imputed if not null, padded, or masked"

        assert torch.all(inputs.time_of_day.lt(1.0), dim=None), "values should be less than 1.0"

        assert torch.all(inputs.time_of_day.ge(0.0), dim=None), "values should be greater than or equal to 0.0"

        N, L = inputs.time_of_day.shape

        # weight inputs with buffers of precision bands
        weighted = inputs.time_of_day.view(N * L).unsqueeze(dim=1).mul(self.weights)

        # apply sine and cosine functions to weighted inputs
        fourier = torch.sin(weighted), torch.cos(weighted)

        # project sinusoidal representations with MLP
        projections = self.linear(torch.cat(fourier, dim=1)).view(N, L, -1)

        projections += self.embeddings["week_of_year"](inputs.week_of_year)
        projections += self.embeddings["day_of_week"](inputs.day_of_week)

        return projections


FieldEmbedder: TypeAlias = DiscreteFieldEmbedder | EntityFieldEmbedder | ContinuousFieldEmbedder | TemporalFieldEmbedder