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
    lookup: Int[torch.Tensor, "_N L"]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(
        cls, values: list[str | None], field: FieldRequest, params: Hyperparameters, manager: SystemManager
    ) -> TensorField:
        mapping = manager.levelsets[field]

        tokens = list(map(lambda value: mapping[value], values))

        padding = (params.n_context - len(tokens), 0)

        array = np.array(tokens, dtype=int)
        lookup = pad(torch.tensor(array), pad=padding, value=Tokens.PAD).unsqueeze(0)

        return cls(lookup=lookup, batch_size=[1])

    @jaxtyped(typechecker=beartype)
    def mask(self, is_event_masked: torch.Tensor, params: Hyperparameters) -> TargetField:
        N, L = (1, params.n_context)
        mask_token = torch.full((N, L), Tokens.MASK)

        is_field_masked = torch.rand(N, L).lt(params.p_mask_field)
        is_masked = is_field_masked.logical_or(is_event_masked)

        targets = self.lookup.clone()

        self.lookup = self.lookup.masked_scatter(is_masked, mask_token)

        return TargetField(lookup=targets, is_masked=is_masked, batch_size=self.batch_size)  # type: ignore

    def prune(self):
        self.lookup = torch.full_like(self.lookup, Tokens.PRUNE)

    @classmethod
    def get_target_size(cls, params: Hyperparameters, manager: SystemManager, field: FieldRequest) -> int:
        return manager.levelsets.get_size(field) + len(Tokens)

    @classmethod
    def postprocess(cls, values: torch.Tensor, targets: torch.Tensor, params: Hyperparameters) -> torch.Tensor:
        N, L, T = values.shape
        return torch.nn.functional.one_hot(targets.lookup.reshape(N * L), num_classes=T).float()

    @classmethod
    def mock(cls, field: FieldRequest, params: Hyperparameters, manager: SystemManager) -> TensorField:
        mappings = manager.levelsets.mappings[field.name]

        levels = list(mappings.keys())

        L = params.n_context

        values = random.choices(levels, k=L)

        return cls.new(values=values, field=field, params=params, manager=manager)


@FieldInterface.register(FieldType.Category)
@tensorclass
class StaticCategoryField(TensorField):
    lookup: Int[torch.Tensor, "_N"]  # noqa: F821

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(
        cls, values: str | None, field: FieldRequest, params: Hyperparameters, manager: SystemManager
    ) -> TensorField:
        mapping = manager.levelsets[field]
        if values is None:
            tokens = Tokens.UNK
        else:
            tokens = mapping[values]

        lookup = torch.tensor([tokens])

        return cls(lookup=lookup, batch_size=[1])

    @jaxtyped(typechecker=beartype)
    def mask(self, is_event_masked: torch.Tensor, params: Hyperparameters) -> TargetField:
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
        N, T = values.shape
        return torch.nn.functional.one_hot(targets.lookup.reshape(N), num_classes=T).float()

    @classmethod
    def mock(cls, field: FieldRequest, params: Hyperparameters, manager: SystemManager) -> TensorField:
        mappings = manager.levelsets.mappings[field.name]

        levels = list(mappings.keys())

        value = random.choice(levels)

        return cls.new(values=value, field=field, params=params, manager=manager)


@FieldInterface.register(FieldType.Entities)
@tensorclass
class DynamicEntityField(TensorField):
    lookup: Int[torch.Tensor, "_N L"]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(
        cls, values: list[str | None], field: FieldRequest, params: Hyperparameters, manager: SystemManager
    ) -> TensorField:
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
        return params.n_context + len(Tokens)

    @classmethod
    def mock(cls, field: FieldRequest, params: Hyperparameters, manager: SystemManager) -> TensorField:
        L = params.n_context

        letters = string.ascii_letters
        values = ["".join(random.choices(letters, k=2)) for _ in range(L)]

        return cls.new(values=values, field=field, params=params, manager=manager)


@FieldInterface.register(FieldType.Numbers)
@tensorclass
class DynamicNumberField(TensorField):
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
        digest: TDigest = manager.centroids.digests[field.name]
        array = np.array(values, dtype=np.float64)
        cdfs = digest.cdf(array)

        lookup = np.where(np.isnan(cdfs), Tokens.UNK, Tokens.VAL)
        content = np.nan_to_num(np.array(cdfs, dtype=float))

        padding = (params.n_context - len(content), 0)

        # this effectively creates `1-(1/inf)` to prevent an index error
        # somewhere in the dataset this exists a CDF of `1.0`, which will not be "floored" correctly
        dampener = 1 - torch.finfo(torch.half).tiny

        return cls(
            content=pad(torch.tensor(content), pad=padding, value=0.0).float().unsqueeze(0).mul(dampener),
            lookup=pad(torch.tensor(lookup), pad=padding, value=Tokens.PAD).unsqueeze(0),
            batch_size=[1],
        )

    @jaxtyped(typechecker=beartype)
    def mask(self, is_event_masked: torch.Tensor, params: Hyperparameters) -> TargetField:
        N, L = (1, params.n_context)
        mask_token = torch.full((N, L), Tokens.MASK)

        # fine out what to mask
        is_field_masked = torch.rand(N, L).lt(params.p_mask_field)
        is_masked = is_field_masked | is_event_masked

        # creating discrete targets
        quantiles = self.content.mul(params.n_quantiles).floor().long().add(len(Tokens))
        targets = self.lookup.masked_scatter(self.lookup == Tokens.VAL, quantiles)

        # mask original values
        self.lookup = self.lookup.masked_scatter(is_masked, mask_token)
        self.content *= ~is_masked

        # return SSL target
        return TargetField(lookup=targets, is_masked=is_masked, batch_size=self.batch_size)  # type: ignore

    def prune(self):
        self.lookup = torch.full_like(self.lookup, Tokens.PRUNE)
        self.content = torch.zeros_like(self.content)

    @classmethod
    def get_target_size(cls, params: Hyperparameters, manager: SystemManager, field: FieldRequest) -> int:
        return params.n_quantiles + len(Tokens)

    @classmethod
    def postprocess(cls, values: torch.Tensor, targets: torch.Tensor, params: Hyperparameters) -> torch.Tensor:
        N, L, T = values.shape
        return smoothen(targets=targets.lookup, size=params.n_quantiles, sigma=params.quantile_smoothing)

    @classmethod
    def mock(cls, field: FieldRequest, params: Hyperparameters, manager: SystemManager) -> TensorField:
        L = params.n_context

        values = [random.uniform(-100, 100) for _ in range(L)]

        return cls.new(values=values, field=field, params=params, manager=manager)


@FieldInterface.register(FieldType.Number)
@tensorclass
class StaticNumberField(TensorField):
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
        N, T = values.shape

        return smoothen(targets=targets.lookup, size=params.n_quantiles, sigma=params.quantile_smoothing)

    @classmethod
    def mock(cls, field: FieldRequest, params: Hyperparameters, manager: SystemManager) -> TensorField:
        value = random.uniform(-100, 100)

        return cls.new(values=value, field=field, params=params, manager=manager)


@FieldInterface.register(FieldType.Quantiles)
@tensorclass
class DynamicQuantileField(TensorField):
    lookup: Int[torch.Tensor, "_N L"]
    content: Float[torch.Tensor, "_N L"]

    @classmethod
    def get_target_size(cls, params: Hyperparameters, manager: SystemManager, field: FieldRequest) -> int:
        return params.n_quantiles + len(Tokens)

    new = DynamicNumberField.new
    mask = DynamicNumberField.mask
    prune = DynamicNumberField.prune
    postprocess = DynamicNumberField.postprocess
    mock = DynamicNumberField.mock


@FieldInterface.register(FieldType.Quantile)
@tensorclass
class StaticQuantileField(TensorField):
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
        array = np.array(values, dtype="datetime64")
        padding = (params.n_context - len(array), 0)

        is_nan = np.isnan(array)

        lookup = np.where(is_nan, Tokens.UNK, Tokens.VAL)
        week_of_year = ((array.astype("datetime64[D]") - array.astype("datetime64[Y]")) / 7) + len(Tokens)
        day_of_week = ((array.view("int64") - 4) % 7) + len(Tokens)
        time_of_day = (array.astype("datetime64[s]") - array.astype("datetime64[D]")).astype(np.int64) * (
            1 / (1440 * 60)
        )
        hour_of_year = (array.astype("datetime64[h]") - array.astype("datetime64[D]")) + len(Tokens)

        np.putmask(week_of_year, is_nan, Tokens.UNK)
        np.putmask(day_of_week, is_nan, Tokens.UNK)
        np.putmask(time_of_day, is_nan, 0.0)
        np.putmask(hour_of_year, is_nan, Tokens.UNK)

        lookup = pad(torch.tensor(lookup), pad=padding, value=Tokens.PAD).unsqueeze(0)
        week_of_year = pad(torch.tensor(week_of_year.astype(np.int64)), pad=padding, value=Tokens.PAD).unsqueeze(0)
        day_of_week = pad(torch.tensor(day_of_week.astype(np.int64)), pad=padding, value=Tokens.PAD).unsqueeze(0)
        time_of_day = pad(torch.tensor(time_of_day.astype(np.float32)), pad=padding, value=0.0).unsqueeze(0)
        hour_of_year = pad(torch.tensor(hour_of_year.astype(np.int64)), pad=padding, value=Tokens.PAD).unsqueeze(0)

        # print(lookup.numpy())
        # print(time_of_day.numpy())

        # print(time_of_day.mul(lookup).eq(0.0).float().sum())

        # if not torch.all(time_of_day.mul(lookup).eq(0.0)):
        #     print(lookup)
        #     print(time_of_day)

        return cls(
            lookup=lookup,
            week_of_year=week_of_year,
            day_of_week=day_of_week,
            time_of_day=time_of_day,
            hour_of_year=hour_of_year,
            batch_size=[1],
        )

    @jaxtyped(typechecker=beartype)
    def mask(self, is_event_masked: torch.Tensor, params: Hyperparameters) -> TargetField:
        N, L = (1, params.n_context)

        mask_token = torch.full((N, L), Tokens.MASK)

        # fine out what to mask
        is_field_masked = torch.rand(N, L).lt(params.p_mask_field)
        is_masked = is_field_masked | is_event_masked

        # creating discrete targets
        timespan = self.hour_of_year

        targets = self.lookup.masked_scatter(self.lookup == Tokens.VAL, timespan)

        # mask original values
        self.lookup = self.lookup.masked_scatter(is_masked, mask_token)
        self.week_of_year = self.week_of_year.masked_scatter(is_masked, mask_token)
        self.day_of_week = self.day_of_week.masked_scatter(is_masked, mask_token)
        self.time_of_day *= ~is_masked

        # return SSL target
        return TargetField(lookup=targets, is_masked=is_masked, batch_size=self.batch_size)  # type: ignore

    def prune(self):
        self.lookup = torch.full_like(self.lookup, Tokens.PRUNE)
        self.week_of_year = torch.full_like(self.week_of_year, Tokens.PRUNE)
        self.day_of_week = torch.full_like(self.day_of_week, Tokens.PRUNE)
        self.time_of_day = torch.zeros_like(self.time_of_day)

    @classmethod
    def get_target_size(cls, params: Hyperparameters, manager: SystemManager, field: FieldRequest) -> int:
        return 366 * 24 + len(Tokens)

    @classmethod
    def postprocess(cls, values: torch.Tensor, targets: torch.Tensor, params: Hyperparameters) -> torch.Tensor:
        N, L, T = values.shape
        return smoothen(targets=targets.lookup, size=(366 * 24), sigma=params.quantile_smoothing)

    @classmethod
    def mock(cls, field: FieldRequest, params: Hyperparameters, manager: SystemManager) -> TensorField:
        start_date = datetime.datetime(2000, 1, 1)
        end_date = datetime.datetime(2020, 12, 31)

        delta = end_date - start_date
        delta_seconds = int(delta.total_seconds())

        L = params.n_context

        values = [start_date + datetime.timedelta(seconds=random.randint(0, delta_seconds)) for _ in range(L)]

        return cls.new(values=values, field=field, params=params, manager=manager)
