from typing import TypeAlias

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from tensordict.prototype import tensorclass
from torch import Tensor
from torch.nn.functional import pad

from windmark.core.constructs.general import Hyperparameters, Tokens, FieldRequest

from windmark.core.architecture.processors import smoothen
from windmark.core.managers import SystemManager


@tensorclass
class TargetField:
    lookup: Int[Tensor, "N L"]
    is_masked: Bool[Tensor, "N L"]


@tensorclass
class DiscreteField:
    lookup: Int[Tensor, "N L"]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(cls, values: list[int], params: Hyperparameters) -> "DiscreteField":
        padding = (params.n_context - len(values), 0)

        array = np.array(values, dtype=int)
        lookup = pad(torch.tensor(array), pad=padding, value=Tokens.PAD).unsqueeze(0)

        return cls(lookup=lookup, batch_size=[1])

    @jaxtyped(typechecker=beartype)
    def mask(self, is_event_masked: Tensor, params: Hyperparameters) -> TargetField:
        N, L = (1, params.n_context)
        mask_token = torch.full((N, L), Tokens.MASK)

        is_field_masked = torch.rand(N, L).lt(params.p_mask_field)
        is_masked = is_field_masked.logical_or(is_event_masked)

        targets = self.lookup.clone()

        self.lookup = self.lookup.masked_scatter(is_masked, mask_token)

        return TargetField(lookup=targets, is_masked=is_masked, batch_size=self.batch_size)  # type: ignore

    def ablate(self):
        self.lookup = torch.full_like(self.lookup, Tokens.ABLATE)

    @classmethod
    def get_target_size(cls, params: Hyperparameters, manager: SystemManager, field: FieldRequest) -> int:
        return manager.levelsets.get_size(field) + len(Tokens)

    @classmethod
    def postprocess(cls, values, targets, params) -> torch.Tensor:
        N, L, T = values.shape
        return torch.nn.functional.one_hot(targets.lookup.reshape(N * L), num_classes=T).float()


@tensorclass
class EntityField:
    lookup: Int[Tensor, "N L"]

    new = DiscreteField.new
    mask = DiscreteField.mask
    ablate = DiscreteField.ablate
    postprocess = DiscreteField.postprocess

    @classmethod
    def get_target_size(cls, params: Hyperparameters, manager: SystemManager, field: FieldRequest) -> int:
        return params.n_context + len(Tokens)


@tensorclass
class ContinuousField:
    lookup: Int[Tensor, "N L"]
    content: Float[Tensor, "N L"]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(cls, values, params: Hyperparameters) -> "ContinuousField":
        lookup = np.where(np.isnan(values), Tokens.UNK, Tokens.VAL)
        values = np.nan_to_num(np.array(values, dtype=float))

        padding = (params.n_context - len(values), 0)

        # this effectively creates `1-(1/inf)` to prevent an index error
        # somewhere in the dataset this exists a CDF of `1.0`, which will not be "floored" correctly
        dampener = 1 - torch.finfo(torch.half).tiny

        return cls(
            content=pad(torch.tensor(values), pad=padding, value=0.0).float().unsqueeze(0).mul(dampener),
            lookup=pad(torch.tensor(lookup), pad=padding, value=Tokens.PAD).unsqueeze(0),
            batch_size=[1],
        )

    @jaxtyped(typechecker=beartype)
    def mask(self, is_event_masked: Tensor, params: Hyperparameters) -> TargetField:
        N, L = (1, params.n_context)
        mask_token = torch.full((N, L), Tokens.MASK)

        # fine out what to mask
        is_field_masked = torch.rand(N, L).lt(params.p_mask_field)
        is_masked = is_field_masked | is_event_masked

        # creating discrete targets
        quantiles = self.content.mul(params.n_quantiles).floor().long().add(len(Tokens))
        is_not_valued = self.lookup != 0
        targets = quantiles.masked_scatter(is_not_valued, quantiles)

        # mask original values
        self.lookup = self.lookup.masked_scatter(is_masked, mask_token)
        self.content *= ~is_masked

        # return SSL target
        return TargetField(lookup=targets, is_masked=is_masked, batch_size=self.batch_size)  # type: ignore

    def ablate(self):
        self.lookup = torch.full_like(self.lookup, Tokens.ABLATE)
        self.content = torch.zeros_like(self.content)

    @classmethod
    def get_target_size(cls, params: Hyperparameters, manager: SystemManager, field: FieldRequest) -> int:
        return params.n_quantiles + len(Tokens)

    @classmethod
    def postprocess(cls, values, targets, params) -> torch.Tensor:
        N, L, T = values.shape
        return smoothen(targets=targets.lookup, size=params.n_quantiles, sigma=params.quantile_smoothing)


@tensorclass
class TemporalField:
    lookup: Int[Tensor, "N L"]
    week_of_year: Int[Tensor, "N L"]
    day_of_week: Int[Tensor, "N L"]
    hour_of_year: Int[Tensor, "N L"]
    time_of_day: Float[Tensor, "N L"]

    @jaxtyped(typechecker=beartype)
    @classmethod
    def new(cls, values, params: Hyperparameters) -> "TemporalField":
        padding = (params.n_context - len(values), 0)

        lookup = np.where(np.isnan(values), Tokens.UNK, Tokens.VAL)
        week_of_year = np.nan_to_num(
            ((values.astype("datetime64[D]") - values.astype("datetime64[Y]")) / 7) + len(Tokens), nan=Tokens.UNK
        )
        day_of_week = np.nan_to_num((((values.view("int64") - 4) % 7) + len(Tokens)), nan=Tokens.UNK)
        time_of_day = np.nan_to_num(
            (values.astype("datetime64[s]") - values.astype("datetime64[D]")).astype(np.int64)
        ) * (1 / (1440 * 60))

        hour_of_year = np.nan_to_num(
            (values.astype("datetime64[h]") - values.astype("datetime64[D]")) + len(Tokens), nan=Tokens.UNK
        )

        return cls(
            lookup=pad(torch.tensor(lookup), pad=padding, value=Tokens.PAD).unsqueeze(0),
            week_of_year=pad(torch.tensor(week_of_year.astype(np.int64)), pad=padding, value=Tokens.PAD).unsqueeze(0),
            day_of_week=pad(torch.tensor(day_of_week.astype(np.int64)), pad=padding, value=Tokens.PAD).unsqueeze(0),
            time_of_day=pad(torch.tensor(time_of_day.astype(np.float32)), pad=padding, value=0.0).unsqueeze(0),
            hour_of_year=pad(torch.tensor(hour_of_year.astype(np.int64)), pad=padding, value=Tokens.PAD).unsqueeze(0),
            batch_size=[1],
        )

    @jaxtyped(typechecker=beartype)
    def mask(self, is_event_masked: Tensor, params: Hyperparameters) -> TargetField:
        N, L = (1, params.n_context)

        mask_token = torch.full((N, L), Tokens.MASK)

        # fine out what to mask
        is_field_masked = torch.rand(N, L).lt(params.p_mask_field)
        is_masked = is_field_masked | is_event_masked

        # creating discrete targets
        timespan = self.hour_of_year

        is_not_valued = self.lookup != 0
        targets = timespan.masked_scatter(is_not_valued, timespan)

        # mask original values
        self.lookup = self.lookup.masked_scatter(is_masked, mask_token)
        self.week_of_year = self.week_of_year.masked_scatter(is_masked, mask_token)
        self.day_of_week = self.day_of_week.masked_scatter(is_masked, mask_token)
        self.time_of_day *= ~is_masked

        # return SSL target
        return TargetField(lookup=targets, is_masked=is_masked, batch_size=self.batch_size)  # type: ignore

    def ablate(self):
        self.lookup = torch.full_like(self.lookup, Tokens.ABLATE)
        self.week_of_year = torch.full_like(self.week_of_year, Tokens.ABLATE)
        self.day_of_week = torch.full_like(self.day_of_week, Tokens.ABLATE)
        self.time_of_day = torch.zeros_like(self.time_of_day)

    @classmethod
    def get_target_size(cls, params: Hyperparameters, manager: SystemManager, field: FieldRequest) -> int:
        return 366 * 24 + len(Tokens)

    @classmethod
    def postprocess(cls, values, targets, params) -> torch.Tensor:
        N, L, T = values.shape
        return smoothen(targets=targets.lookup, size=(366 * 24), sigma=params.quantile_smoothing)


TensorField: TypeAlias = ContinuousField | DiscreteField | EntityField | TemporalField
