import string
import random
import datetime
from decimal import Decimal

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from tensordict.prototype import tensorclass
from pytdigest import TDigest
from torch.nn.functional import pad

from windmark.core.constructs.general import Hyperparameters, Tokens, FieldRequest, FieldType
from windmark.core.architecture.custom import smoothen
from windmark.core.managers import SystemManager
from windmark.core.constructs.interface import TargetField, TensorField, FieldInterface


@FieldInterface.register(FieldType.Categories)
@tensorclass
class DynamicCategoryField(TensorField):
    """
    Represents a dynamic category field in a tensor field.

    Attributes:
        lookup (Int[torch.Tensor, "_N L"]): The lookup tensor for the dynamic category field.

    Methods:
        new(cls, values, field, params, manager): Creates a new instance of DynamicCategoryField.
        mask(self, is_event_masked, params): Masks the dynamic category field.
        prune(self): Prunes the dynamic category field.
        get_target_size(cls, params, manager, field): Returns the target size of the dynamic category field.
        postprocess(cls, values, targets, params): Postprocesses the dynamic category field.
        mock(cls, field, params, manager): Creates a mock instance of DynamicCategoryField.
    """

    lookup: Int[torch.Tensor, "_N L"]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(
        cls, values: list[str | None], field: FieldRequest, params: Hyperparameters, manager: SystemManager
    ) -> TensorField:
        """
        Creates a new instance of DynamicCategoryField.

        Args:
            values (list[str | None]): The values for the dynamic category field.
            field (FieldRequest): The field request.
            params (Hyperparameters): The hyperparameters.
            manager (SystemManager): The system manager.

        Returns:
            TensorField: The new instance of DynamicCategoryField.
        """
        mapping = manager.levelsets[field]

        tokens = list(map(lambda value: mapping[value], values))

        padding = (params.n_context - len(tokens), 0)

        array = np.array(tokens, dtype=int)
        lookup = pad(torch.tensor(array), pad=padding, value=Tokens.PAD).unsqueeze(0)

        return cls(lookup=lookup, batch_size=[1])

    @jaxtyped(typechecker=beartype)
    def mask(self, is_event_masked: torch.Tensor, params: Hyperparameters) -> TargetField:
        """
        Masks the dynamic category field.

        Args:
            is_event_masked (torch.Tensor): The event mask.
            params (Hyperparameters): The hyperparameters.

        Returns:
            TargetField: The masked dynamic category field.
        """
        N, L = (1, params.n_context)
        mask_token = torch.full((N, L), Tokens.MASK)

        is_field_masked = torch.rand(N, L).lt(params.p_mask_field)
        is_masked = is_field_masked.logical_or(is_event_masked)

        targets = self.lookup.clone()

        self.lookup = self.lookup.masked_scatter(is_masked, mask_token)

        return TargetField(lookup=targets, is_masked=is_masked, batch_size=self.batch_size)  # type: ignore

    def prune(self):
        """
        Prunes the dynamic category field.
        """
        self.lookup = torch.full_like(self.lookup, Tokens.PRUNE)

    @classmethod
    def get_target_size(cls, params: Hyperparameters, manager: SystemManager, field: FieldRequest) -> int:
        """
        Returns the target size of the dynamic category field.

        Args:
            params (Hyperparameters): The hyperparameters.
            manager (SystemManager): The system manager.
            field (FieldRequest): The field request.

        Returns:
            int: The target size of the dynamic category field.
        """
        return manager.levelsets.get_size(field) + len(Tokens)

    @classmethod
    def postprocess(cls, values: torch.Tensor, targets: torch.Tensor, params: Hyperparameters) -> torch.Tensor:
        """
        Postprocesses the dynamic category field.

        Args:
            values (torch.Tensor): The values tensor.
            targets (torch.Tensor): The targets tensor.
            params (Hyperparameters): The hyperparameters.

        Returns:
            torch.Tensor: The postprocessed dynamic category field.
        """
        N, L, T = values.shape
        return torch.nn.functional.one_hot(targets.lookup.reshape(N * L), num_classes=T).float()

    @classmethod
    def mock(cls, field: FieldRequest, params: Hyperparameters, manager: SystemManager) -> TensorField:
        """
        Creates a mock instance of DynamicCategoryField.

        Args:
            field (FieldRequest): The field request.
            params (Hyperparameters): The hyperparameters.
            manager (SystemManager): The system manager.

        Returns:
            TensorField: The mock instance of DynamicCategoryField.
        """
        mappings = manager.levelsets.mappings[field.name]

        levels = list(mappings.keys())

        L = params.n_context

        values = random.choices(levels, k=L)

        return cls.new(values=values, field=field, params=params, manager=manager)


@FieldInterface.register(FieldType.Category)
@tensorclass
class StaticCategoryField(TensorField):
    """
    Represents a static category field in a tensor field.

    Attributes:
        lookup (Int[torch.Tensor, "_N"]): The lookup tensor for the field.

    Methods:
        new(cls, values, field, params, manager): Creates a new instance of StaticCategoryField.
        mask(self, is_event_masked, params): Masks the field based on the given mask tensor.
        postprocess(cls, values, targets, params): Postprocesses the field values and targets.
        mock(cls, field, params, manager): Creates a mock instance of StaticCategoryField.

    """

    lookup: Int[torch.Tensor, "_N"]  # noqa: F821

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(
        cls, values: str | None, field: FieldRequest, params: Hyperparameters, manager: SystemManager
    ) -> TensorField:
        """
        Creates a new instance of StaticCategoryField.

        Args:
            values (str | None): The values for the field.
            field (FieldRequest): The field request.
            params (Hyperparameters): The hyperparameters.
            manager (SystemManager): The system manager.

        Returns:
            TensorField: The new instance of StaticCategoryField.

        """
        mapping = manager.levelsets[field]
        if values is None:
            tokens = Tokens.UNK
        else:
            tokens = mapping[values]

        lookup = torch.tensor([tokens])

        return cls(lookup=lookup, batch_size=[1])

    @jaxtyped(typechecker=beartype)
    def mask(self, is_event_masked: torch.Tensor, params: Hyperparameters) -> TargetField:
        """
        Masks the field based on the given mask tensor.

        Args:
            is_event_masked (torch.Tensor): The mask tensor.
            params (Hyperparameters): The hyperparameters.

        Returns:
            TargetField: The masked target field.

        """
        _ = is_event_masked

        N = 1

        mask_token = torch.full((N,), Tokens.MASK)

        is_field_masked = torch.rand(1).lt(params.p_mask_static)

        targets = self.lookup.clone()

        self.lookup = self.lookup.masked_scatter(is_field_masked, mask_token)

        return TargetField(lookup=targets, is_masked=is_field_masked, batch_size=self.batch_size)  # type: ignore

    prune = DynamicCategoryField.prune
    get_target_size = DynamicCategoryField.get_target_size

    @classmethod
    def postprocess(cls, values: torch.Tensor, targets: torch.Tensor, params: Hyperparameters) -> torch.Tensor:
        """
        Postprocesses the field values and targets.

        Args:
            values (torch.Tensor): The field values.
            targets (torch.Tensor): The field targets.
            params (Hyperparameters): The hyperparameters.

        Returns:
            torch.Tensor: The postprocessed tensor.

        """
        N, T = values.shape
        return torch.nn.functional.one_hot(targets.lookup.reshape(N), num_classes=T).float()

    @classmethod
    def mock(cls, field: FieldRequest, params: Hyperparameters, manager: SystemManager) -> TensorField:
        """
        Creates a mock instance of StaticCategoryField.

        Args:
            field (FieldRequest): The field request.
            params (Hyperparameters): The hyperparameters.
            manager (SystemManager): The system manager.

        Returns:
            TensorField: The mock instance of StaticCategoryField.

        """
        mappings = manager.levelsets.mappings[field.name]

        levels = list(mappings.keys())

        value = random.choice(levels)

        return cls.new(values=value, field=field, params=params, manager=manager)


@FieldInterface.register(FieldType.Entities)
@tensorclass
class DynamicEntityField(TensorField):
    """
    Represents a dynamic entity field in a tensor field.

    Inherits from the TensorField class.
    """

    lookup: Int[torch.Tensor, "_N L"]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(
        cls, values: list[str | None], field: FieldRequest, params: Hyperparameters, manager: SystemManager
    ) -> TensorField:
        """
        Creates a new instance of the DynamicEntityField class.

        Args:
            values (list[str | None]): The values for the field.
            field (FieldRequest): The field request.
            params (Hyperparameters): The hyperparameters.
            manager (SystemManager): The system manager.

        Returns:
            TensorField: The new instance of the DynamicEntityField class.
        """

        unique: set[str] = set(values)

        integers = random.sample(range(len(Tokens), params.n_context + len(Tokens)), len(unique))

        mapping = dict(zip(unique, integers))

        mapping.update({None: Tokens.UNK})

        tokens = list(map(lambda value: mapping[value], values))

        padding = (params.n_context - len(tokens), 0)

        array = np.array(tokens, dtype=int)
        lookup = pad(torch.tensor(array), pad=padding, value=Tokens.PAD).unsqueeze(0)

        return cls(lookup=lookup, batch_size=[1])

    mask = DynamicCategoryField.mask
    prune = DynamicCategoryField.prune
    postprocess = DynamicCategoryField.postprocess

    @classmethod
    def get_target_size(cls, params: Hyperparameters, manager: SystemManager, field: FieldRequest) -> int:
        """
        Gets the target size of the dynamic entity field.

        Args:
            params (Hyperparameters): The hyperparameters.
            manager (SystemManager): The system manager.
            field (FieldRequest): The field request.

        Returns:
            int: The target size of the dynamic entity field.
        """
        return params.n_context + len(Tokens)

    @classmethod
    def mock(cls, field: FieldRequest, params: Hyperparameters, manager: SystemManager) -> TensorField:
        """
        Creates a mock instance of the DynamicEntityField class.

        Args:
            field (FieldRequest): The field request.
            params (Hyperparameters): The hyperparameters.
            manager (SystemManager): The system manager.

        Returns:
            TensorField: The mock instance of the DynamicEntityField class.
        """
        L = params.n_context

        letters = string.ascii_letters
        values = ["".join(random.choices(letters, k=2)) for _ in range(L)]

        return cls.new(values=values, field=field, params=params, manager=manager)


@FieldInterface.register(FieldType.Numbers)
@tensorclass
class DynamicNumberField(TensorField):
    """
    A class representing a dynamic number field in a tensor field.

    Attributes:
        lookup (Int[torch.Tensor, "_N L"]): The lookup tensor.
        content (Float[torch.Tensor, "_N L"]): The content tensor.

    Methods:
        new(cls, values, field, params, manager): Creates a new instance of DynamicNumberField.
        mask(self, is_event_masked, params): Applies a mask to the field.
        prune(self): Prunes the field.
        get_target_size(cls, params, manager, field): Returns the target size.
        postprocess(cls, values, targets, params): Performs postprocessing on the field.
        mock(cls, field, params, manager): Creates a mock instance of DynamicNumberField.
    """

    lookup: Int[torch.Tensor, "_N L"]
    content: Float[torch.Tensor, "_N L"]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(
        cls,
        values: list[Decimal | int | float | str | None],
        field: FieldRequest,
        params: Hyperparameters,
        manager: SystemManager,
    ) -> TensorField:
        """
        Creates a new instance of DynamicNumberField.

        Args:
            values (list[Decimal | int | float | str | None]): The values for the field.
            field (FieldRequest): The field request.
            params (Hyperparameters): The hyperparameters.
            manager (SystemManager): The system manager.

        Returns:
            TensorField: The new instance of DynamicNumberField.
        """
        # Code implementation...

    @jaxtyped(typechecker=beartype)
    def mask(self, is_event_masked: torch.Tensor, params: Hyperparameters) -> TargetField:
        """
        Applies a mask to the field.

        Args:
            is_event_masked (torch.Tensor): The event mask.
            params (Hyperparameters): The hyperparameters.

        Returns:
            TargetField: The masked field.
        """
        # Code implementation...

    def prune(self):
        """
        Prunes the field.
        """
        # Code implementation...

    @classmethod
    def get_target_size(cls, params: Hyperparameters, manager: SystemManager, field: FieldRequest) -> int:
        """
        Returns the target size.

        Args:
            params (Hyperparameters): The hyperparameters.
            manager (SystemManager): The system manager.
            field (FieldRequest): The field request.

        Returns:
            int: The target size.
        """
        # Code implementation...

    @classmethod
    def postprocess(cls, values: torch.Tensor, targets: torch.Tensor, params: Hyperparameters) -> torch.Tensor:
        """
        Performs postprocessing on the field.

        Args:
            values (torch.Tensor): The values tensor.
            targets (torch.Tensor): The targets tensor.
            params (Hyperparameters): The hyperparameters.

        Returns:
            torch.Tensor: The processed tensor.
        """
        # Code implementation...

    @classmethod
    def mock(cls, field: FieldRequest, params: Hyperparameters, manager: SystemManager) -> TensorField:
        """
        Creates a mock instance of DynamicNumberField.

        Args:
            field (FieldRequest): The field request.
            params (Hyperparameters): The hyperparameters.
            manager (SystemManager): The system manager.

        Returns:
            TensorField: The mock instance of DynamicNumberField.
        """
        # Code implementation...


@FieldInterface.register(FieldType.Number)
@tensorclass
class StaticNumberField(TensorField):
    """
    Represents a static number field in a tensor field.
    """

    lookup: Int[torch.Tensor, "_N"]  # noqa: F821
    content: Float[torch.Tensor, "_N"]  # noqa: F821

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(
        cls,
        values: str | Decimal | int | float | None,
        field: FieldRequest,
        params: Hyperparameters,
        manager: SystemManager,
    ) -> TensorField:
        """
        Creates a new instance of the StaticNumberField class.

        Args:
            values: The values for the field.
            field: The field request.
            params: The hyperparameters.
            manager: The system manager.

        Returns:
            A new instance of the StaticNumberField class.
        """
        digest: TDigest = manager.centroids.digests[field.name]

        if values is None:
            lookup: int = Tokens.UNK
            content: float = 0.0

        else:
            lookup: int = Tokens.VAL
            content: float = digest.cdf(values)

        # this effectively creates `1-(1/inf)` to prevent an index error
        # somewhere in the dataset this exists a CDF of `1.0`, which will not be "floored" correctly
        dampener = 1 - torch.finfo(torch.half).tiny

        return cls(
            content=torch.tensor([content]).unsqueeze(0).mul(dampener),
            lookup=torch.tensor([lookup]).unsqueeze(0),
            batch_size=[1],
        )

    @jaxtyped(typechecker=beartype)
    def mask(self, is_event_masked: torch.Tensor, params: Hyperparameters) -> TargetField:
        """
        Masks the static number field.

        Args:
            is_event_masked: The tensor indicating whether an event is masked.
            params: The hyperparameters.

        Returns:
            The masked target field.
        """
        _ = is_event_masked
        N = 1
        mask_token = torch.full((N,), Tokens.MASK)

        # fine out what to mask
        is_field_masked = torch.rand(N).lt(params.p_mask_static)

        # creating discrete targets
        quantiles = self.content.mul(params.n_quantiles).floor().long().add(len(Tokens))
        targets = self.lookup.masked_scatter(self.lookup == Tokens.VAL, quantiles)

        # mask original values
        self.lookup = self.lookup.masked_scatter(is_field_masked, mask_token)
        self.content *= ~is_field_masked

        # return SSL target
        return TargetField(lookup=targets, is_masked=is_field_masked, batch_size=self.batch_size)  # type: ignore

    prune = DynamicNumberField.prune
    get_target_size = DynamicNumberField.get_target_size

    @classmethod
    def postprocess(cls, values: torch.Tensor, targets: torch.Tensor, params: Hyperparameters) -> torch.Tensor:
        """
        Postprocesses the static number field.

        Args:
            values: The tensor of values.
            targets: The tensor of targets.
            params: The hyperparameters.

        Returns:
            The postprocessed tensor.
        """
        N, T = values.shape

        return smoothen(targets=targets.lookup, size=params.n_quantiles, sigma=params.quantile_smoothing)

    @classmethod
    def mock(cls, field: FieldRequest, params: Hyperparameters, manager: SystemManager) -> TensorField:
        """
        Creates a mock instance of the StaticNumberField class.

        Args:
            field: The field request.
            params: The hyperparameters.
            manager: The system manager.

        Returns:
            A mock instance of the StaticNumberField class.
        """
        value = random.uniform(-100, 100)

        return cls.new(values=value, field=field, params=params, manager=manager)


@FieldInterface.register(FieldType.Quantiles)
@tensorclass
class DynamicQuantileField(TensorField):
    """
    A class representing a dynamic quantile field.

    Inherits from the `TensorField` class.

    Attributes:
        lookup (Int[torch.Tensor, "_N L"]): The lookup tensor.
        content (Float[torch.Tensor, "_N L"]): The content tensor.

    Methods:
        get_target_size(cls, params: Hyperparameters, manager: SystemManager, field: FieldRequest) -> int:
            Returns the target size of the dynamic quantile field.
    """

    lookup: Int[torch.Tensor, "_N L"]
    content: Float[torch.Tensor, "_N L"]

    @classmethod
    def get_target_size(cls, params: Hyperparameters, manager: SystemManager, field: FieldRequest) -> int:
        """
        Returns the target size of the dynamic quantile field.

        Args:
            params (Hyperparameters): The hyperparameters.
            manager (SystemManager): The system manager.
            field (FieldRequest): The field request.

        Returns:
            int: The target size of the dynamic quantile field.
        """
        return params.n_quantiles + len(Tokens)

    new = DynamicNumberField.new
    mask = DynamicNumberField.mask
    prune = DynamicNumberField.prune
    postprocess = DynamicNumberField.postprocess
    mock = DynamicNumberField.mock


@FieldInterface.register(FieldType.Quantile)
@tensorclass
class StaticQuantileField(TensorField):
    """
    Represents a static quantile field.

    This class inherits from the `TensorField` class and provides functionality specific to static quantile fields.

    Attributes:
        lookup (Int[torch.Tensor, "_N"]): The lookup tensor for the quantile field.
        content (Float[torch.Tensor, "_N"]): The content tensor for the quantile field.
    """

    lookup: Int[torch.Tensor, "_N"]  # noqa: F821
    content: Float[torch.Tensor, "_N"]  # noqa: F821

    get_target_size = DynamicQuantileField.get_target_size

    new = StaticNumberField.new
    mask = StaticNumberField.prune
    prune = StaticNumberField.prune
    postprocess = StaticNumberField.postprocess
    mock = StaticNumberField.mock


@FieldInterface.register(FieldType.Timestamps)
@tensorclass
class DynamicTemporalField(TensorField):
    """
    Represents a dynamic temporal field in a tensor field.

    Attributes:
        lookup (torch.Tensor): The lookup tensor.
        week_of_year (torch.Tensor): The week of year tensor.
        day_of_week (torch.Tensor): The day of week tensor.
        hour_of_year (torch.Tensor): The hour of year tensor.
        time_of_day (torch.Tensor): The time of day tensor.
    """

    lookup: Int[torch.Tensor, "_N L"]
    week_of_year: Int[torch.Tensor, "_N L"]
    day_of_week: Int[torch.Tensor, "_N L"]
    hour_of_year: Int[torch.Tensor, "_N L"]
    time_of_day: Float[torch.Tensor, "_N L"]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(
        cls,
        values: list[datetime.datetime | str | None],
        field: FieldRequest,
        params: Hyperparameters,
        manager: SystemManager,
    ) -> TensorField:
        """
        Creates a new instance of the DynamicTemporalField class.

        Args:
            values (list[datetime.datetime | str | None]): The values for the field.
            field (FieldRequest): The field request.
            params (Hyperparameters): The hyperparameters.
            manager (SystemManager): The system manager.

        Returns:
            TensorField: The created tensor field.
        """
        # Code implementation...

    @jaxtyped(typechecker=beartype)
    def mask(self, is_event_masked: torch.Tensor, params: Hyperparameters) -> TargetField:
        """
        Masks the field based on the given event mask and hyperparameters.

        Args:
            is_event_masked (torch.Tensor): The event mask.
            params (Hyperparameters): The hyperparameters.

        Returns:
            TargetField: The masked target field.
        """
        # Code implementation...

    def prune(self):
        """
        Prunes the field by setting all values to the prune token.
        """
        # Code implementation...

    @classmethod
    def get_target_size(cls, params: Hyperparameters, manager: SystemManager, field: FieldRequest) -> int:
        """
        Gets the target size for the field.

        Args:
            params (Hyperparameters): The hyperparameters.
            manager (SystemManager): The system manager.
            field (FieldRequest): The field request.

        Returns:
            int: The target size.
        """
        # Code implementation...

    @classmethod
    def postprocess(cls, values: torch.Tensor, targets: torch.Tensor, params: Hyperparameters) -> torch.Tensor:
        """
        Postprocesses the values and targets.

        Args:
            values (torch.Tensor): The values tensor.
            targets (torch.Tensor): The targets tensor.
            params (Hyperparameters): The hyperparameters.

        Returns:
            torch.Tensor: The postprocessed tensor.
        """
        # Code implementation...

    @classmethod
    def mock(cls, field: FieldRequest, params: Hyperparameters, manager: SystemManager) -> TensorField:
        """
        Creates a mock instance of the DynamicTemporalField class.

        Args:
            field (FieldRequest): The field request.
            params (Hyperparameters): The hyperparameters.
            manager (SystemManager): The system manager.

        Returns:
            TensorField: The created tensor field.
        """
        # Code implementation...
