import math

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from windmark.core.constructs.general import Hyperparameters, Tokens, FieldRequest, FieldType
from windmark.core.constructs.tensorfields import FieldInterface
from windmark.core.managers import SystemManager
from windmark.core.constructs.interface import FieldEmbedder, TensorField
from windmark.core.architecture.custom import validate, jitter


@FieldInterface.register(FieldType.Categories)
class DynamicCategoryFieldEmbedder(FieldEmbedder):
    """
    Embedder for dynamic categorical fields.

    This class represents an embedder for dynamic categorical fields. It inherits from the `FieldEmbedder` base class.

    Attributes:
        field (FieldRequest): The field to be embedded.
        embeddings (torch.nn.Embedding): The embedding layer for the field.

    Methods:
        __init__(self, params: Hyperparameters, manager: SystemManager, field: FieldRequest): Initializes the embedder.
        forward(self, inputs: TensorField) -> Float[torch.Tensor, "_N L C"]: Performs the forward pass of the embedder.

    """

    def __init__(self, params: Hyperparameters, manager: SystemManager, field: FieldRequest):
        super().__init__()

        self.field: FieldRequest = field
        self.embeddings = torch.nn.Embedding(manager.levelsets.get_size(field) + len(Tokens), params.d_field)

    @jaxtyped(typechecker=beartype)
    def forward(self, inputs: TensorField) -> Float[torch.Tensor, "_N L C"]:
        """
        Perform the forward pass of the embedder.

        Args:
            inputs (TensorField): The input tensor field.

        Returns:
            Float[torch.Tensor, "_N L C"]: The embedded tensor.

        """
        return self.embeddings(inputs.lookup)


@FieldInterface.register(FieldType.Category)
class StaticCategoryFieldEmbedder(FieldEmbedder):
    """
    Embedder for static categorical fields.

    Args:
        params (Hyperparameters): The hyperparameters for the embedder.
        manager (SystemManager): The system manager.
        field (FieldRequest): The field to embed.

    Attributes:
        field (FieldRequest): The field to embed.
        embeddings (torch.nn.Embedding): The embedding layer.

    """

    def __init__(self, params: Hyperparameters, manager: SystemManager, field: FieldRequest):
        super().__init__()

        self.field: FieldRequest = field
        self.embeddings = torch.nn.Embedding(
            manager.levelsets.get_size(field) + len(Tokens), params.d_field * len(manager.schema.dynamic)
        )

    @jaxtyped(typechecker=beartype)
    def forward(self, inputs: TensorField) -> Float[torch.Tensor, "_N FdC"]:
        """
        Forward pass of the embedder.

        Args:
            inputs (TensorField): The input tensor field.

        Returns:
            Float[torch.Tensor, "_N FdC"]: The embedded tensor.

        """
        return self.embeddings(inputs.lookup)


@FieldInterface.register(FieldType.Entities)
class DynamicEntityFieldEmbedder(FieldEmbedder):
    """
    Embedder for dynamic entity fields.

    Args:
        params (Hyperparameters): The hyperparameters for the embedder.
        manager (SystemManager): The system manager.
        field (FieldRequest): The field request.

    Attributes:
        field (FieldRequest): The field request.
        embeddings (torch.nn.Embedding): The embedding layer.

    """

    def __init__(self, params: Hyperparameters, manager: SystemManager, field: FieldRequest):
        super().__init__()

        self.field: FieldRequest = field
        self.embeddings = torch.nn.Embedding(params.n_context + len(Tokens), params.d_field)

    @jaxtyped(typechecker=beartype)
    def forward(self, inputs: TensorField) -> torch.Tensor:
        """
        Forward pass of the embedder.

        Args:
            inputs (TensorField): The input tensor field.

        Returns:
            torch.Tensor: The embedded tensor.

        """
        return self.embeddings(inputs.lookup)


@FieldInterface.register(FieldType.Quantiles)
class DynamicQuantileFieldEmbedder(FieldEmbedder):
    """
    Embedder that maps field values to continuous embeddings using dynamic quantization.

    Args:
        params (Hyperparameters): The hyperparameters for the embedder.
        manager (SystemManager): The system manager.
        field (FieldRequest): The field request.

    Attributes:
        field (FieldRequest): The field request.
        embeddings (torch.nn.Embedding): The embedding layer.
        jitter (torch.Tensor): The jitter value.
        n_quantiles (torch.Tensor): The number of quantiles.
        dampener (torch.Tensor): The dampener value.
    """

    def __init__(self, params: Hyperparameters, manager: SystemManager, field: FieldRequest):
        super().__init__()

        self.field: FieldRequest = field
        self.embeddings = torch.nn.Embedding(params.n_quantiles + len(Tokens), params.d_field)

        self.register_buffer("jitter", torch.tensor(params.jitter))
        self.register_buffer("n_quantiles", torch.tensor(params.n_quantiles))

    @jaxtyped(typechecker=beartype)
    def forward(self, inputs: TensorField) -> Float[torch.Tensor, "_N L C"]:
        """
        Forward pass of the embedder.

        Args:
            inputs (TensorField): The input tensor field.

        Returns:
            Float[torch.Tensor, "_N L C"]: The embedded tensor.
        """

        indicators = inputs.lookup

        validate(inputs)

        jittered = jitter(inputs, jitter=self.jitter, is_training=self.training)

        quantiles = jittered.mul(self.n_quantiles).floor().long().add(len(Tokens))
        lookup = indicators.masked_scatter(indicators == Tokens.VAL, quantiles)

        return self.embeddings(lookup)


@FieldInterface.register(FieldType.Quantile)
class StaticQuantileFieldEmbedder(FieldEmbedder):
    """
    Embedder that maps field values to embeddings using static quantiles.

    Args:
        params (Hyperparameters): The hyperparameters for the embedder.
        manager (SystemManager): The system manager.
        field (FieldRequest): The field request.

    Attributes:
        field (FieldRequest): The field request.
        embeddings (torch.nn.Embedding): The embedding layer.
        jitter (torch.Tensor): The jitter tensor.
        n_quantiles (torch.Tensor): The number of quantiles tensor.
        dampener (torch.Tensor): The dampener tensor.
    """

    def __init__(self, params: Hyperparameters, manager: SystemManager, field: FieldRequest):
        super().__init__()

        self.field: FieldRequest = field
        self.embeddings = torch.nn.Embedding(
            params.n_quantiles + len(Tokens), params.d_field * len(manager.schema.dynamic)
        )

        self.register_buffer("jitter", torch.tensor(params.jitter))
        self.register_buffer("n_quantiles", torch.tensor(params.n_quantiles))

    @jaxtyped(typechecker=beartype)
    def forward(self, inputs: TensorField) -> Float[torch.Tensor, "_N FdC"]:
        """
        Forward pass of the embedder.

        Args:
            inputs (TensorField): The input tensor field.

        Returns:
            Float[torch.Tensor, "_N FdC"]: The embedded tensor.
        """

        indicators = inputs.lookup

        validate(inputs)

        jittered = jitter(inputs, jitter=self.jitter, is_training=self.training)

        quantiles = jittered.mul(self.n_quantiles).floor().long().add(len(Tokens))
        lookup = indicators.masked_scatter(indicators == Tokens.VAL, quantiles)

        return self.embeddings(lookup)


@FieldInterface.register(FieldType.Numbers)
class DynamicNumberFieldEmbedder(FieldEmbedder):
    """
    Embedder for dynamic number fields.

    This class implements the embedding logic for dynamic number fields. It takes a set of hyperparameters,
    a system manager, and a field request as input. It performs embedding operations on the input data and
    returns the embedded representations.

    Args:
        params (Hyperparameters): The hyperparameters for the embedding.
        manager (SystemManager): The system manager for the embedding.
        field (FieldRequest): The field request for the embedding.

    Attributes:
        field (FieldRequest): The field request for the embedding.
        linear (torch.nn.Linear): The linear layer for projection.
        weights (torch.Tensor): The weights for input weighting.
        jitter (torch.Tensor): The jitter for input perturbation.

    """

    def __init__(self, params: Hyperparameters, manager: SystemManager, field: FieldRequest):
        super().__init__()

        self.field: FieldRequest = field

        offset = 4

        weights = torch.logspace(start=-params.n_bands, end=offset, steps=params.n_bands + offset + 1, base=2)

        self.linear = torch.nn.Linear(2 * len(weights), params.d_field)
        self.register_buffer("weights", weights.mul(math.pi).unsqueeze(dim=0))
        self.register_buffer("jitter", torch.tensor(params.jitter))

    @jaxtyped(typechecker=beartype)
    def forward(self, inputs: TensorField) -> Float[torch.Tensor, "_N L C"]:
        """
        Forward pass of the embedding.

        This method performs the forward pass of the embedding. It takes an input tensor field and returns
        the embedded representations.

        Args:
            inputs (TensorField): The input tensor field.

        Returns:
            Float[torch.Tensor, "_N L C"]: The embedded representations.

        """

        values = inputs.content
        indicators = inputs.lookup

        validate(inputs)

        N, L = values.shape

        jittered = jitter(inputs, jitter=self.jitter, is_training=self.training)

        # weight inputs with buffers of precision bands
        weighted = jittered.add(indicators * 2).view(N * L).unsqueeze(dim=1).mul(self.weights)

        # apply sine and cosine functions to weighted inputs
        fourier = torch.sin(weighted), torch.cos(weighted)

        # project sinusoidal representations with MLP
        projections = torch.nn.functional.gelu(self.linear(torch.cat(fourier, dim=1)).view(N, L, -1))

        return projections


@FieldInterface.register(FieldType.Number)
class StaticNumberFieldEmbedder(FieldEmbedder):
    """
    Embedder for static number fields.

    This class represents an embedder for static number fields. It takes in a set of hyperparameters,
    a system manager, and a field request as input. It performs embedding operations on the input data
    and returns the projections.

    Args:
        params (Hyperparameters): The hyperparameters for the embedder.
        manager (SystemManager): The system manager.
        field (FieldRequest): The field request.

    Attributes:
        field (FieldRequest): The field request.
        linear (torch.nn.Linear): The linear layer for projection.
        weights (torch.Tensor): The weights for input weighting.

    """

    def __init__(self, params: Hyperparameters, manager: SystemManager, field: FieldRequest):
        super().__init__()

        self.field: FieldRequest = field

        offset = 3

        weights = torch.logspace(start=-params.n_bands, end=offset, steps=params.n_bands + offset + 1, base=2)

        self.linear = torch.nn.Linear(2 * len(weights), params.d_field * len(manager.schema.dynamic))
        self.register_buffer("weights", weights.mul(math.pi).unsqueeze(dim=0))

    @jaxtyped(typechecker=beartype)
    def forward(self, inputs: TensorField) -> Float[torch.Tensor, "_N FdC"]:
        """
        Forward pass of the embedder.

        This method performs the forward pass of the embedder. It takes in the input tensor field,
        performs embedding operations, and returns the projections.

        Args:
            inputs (TensorField): The input tensor field.

        Returns:
            torch.Tensor: The projections.
        """

        indicators = inputs.lookup

        validate(inputs)

        jittered = jitter(inputs, jitter=self.jitter, is_training=self.training)

        # weight inputs with buffers of precision bands
        weighted = jittered.add(indicators * 2).mul(self.weights)

        # apply sine and cosine functions to weighted inputs
        fourier = torch.cat((torch.sin(weighted), torch.cos(weighted)), dim=1)

        # project sinusoidal representations with MLP
        projections = torch.nn.functional.gelu(self.linear(fourier))

        return projections


@FieldInterface.register(FieldType.Timestamps)
class DynamicTemporalFieldEmbedder(FieldEmbedder):
    """
    Embedder for dynamic temporal fields.

    Args:
        params (Hyperparameters): The hyperparameters for the embedder.
        manager (SystemManager): The system manager.
        field (FieldRequest): The field request.

    Attributes:
        field (FieldRequest): The field request.
        linear (torch.nn.Linear): Linear layer for projection.
        weights (torch.Tensor): Buffer of precision bands.
        embeddings (torch.nn.ModuleDict): Module dictionary for embeddings.

    """

    def __init__(self, params: Hyperparameters, manager: SystemManager, field: FieldRequest):
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
    def forward(self, inputs: TensorField) -> Float[torch.Tensor, "_N L C"]:
        """
        Forward pass of the embedder.

        Args:
            inputs (TensorField): The input tensor field.

        Returns:
            torch.Tensor: The projected embeddings.

        Raises:
            AssertionError: If the shape of time_of_day and lookup tensors are not the same.
            AssertionError: If any value in time_of_day and lookup tensors is not 0.0.
            AssertionError: If any value in time_of_day tensor is not less than 1.0.
            AssertionError: If any value in time_of_day tensor is not greater than or equal to 0.0.

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
        projections = torch.nn.functional.gelu(self.linear(torch.cat(fourier, dim=1)).view(N, L, -1))

        projections += self.embeddings["week_of_year"](inputs.week_of_year)
        projections += self.embeddings["day_of_week"](inputs.day_of_week)

        return projections
