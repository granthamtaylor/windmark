import os
import random
from typing import Sequence
from functools import partial

import fastparquet
import polars as pl
import torch
from tensordict import TensorDict
from torchdata import datapipes
import fastavro
from tdigest import TDigest
import numpy as np
from torch.nn.functional import pad
from msgspec import json
import lightning.pytorch as lit

from source.core.schema import SPECIAL_TOKENS, Hyperparameters, Field


def read(filename):
    with open(filename, "rb") as f:
        reader = fastavro.reader(f)
        records = [record for record in reader]

    return records


def sample(sequence: dict, params: Hyperparameters) -> list[dict[str, int | float | None]]:
    sequence_id = sequence["sequence_id"]
    observations: list[dict[str, int | float | None]] = []

    for event in range(sequence["size"]):
        if (label := sequence["target"][event]) is None:
            continue

        if params.pretrain_sample_rate < random.random():
            continue

        window = slice(max(0, event - params.n_context), event)

        observation = dict(
            sequence_id=str(sequence_id),
            event_id=str(sequence["event_ids"][event]),
            label=label,
        )

        for field in params.fields:
            observation[field.name] = sequence[field.name][window]

        observations.append(observation)

    return observations


def hash(
    observation: dict[str, list[int] | list[float | None] | list[str | None]],
    n_context: int,
    fieldname: str,
    token_map: dict[str, int],
) -> dict[str, list[int] | list[float | None] | list[str | None]]:
    field = observation[fieldname]

    unique = set(field)

    [unique.remove(token) for token in token_map.keys()]

    integers = random.sample(range(len(token_map), n_context + len(token_map)), len(unique))

    mapping = dict(zip(unique, integers))

    mapping.update(token_map)

    observation["fieldname"] = [mapping[token] for token in field]


def cdf(
    observation: dict[str, list[int] | list[float | None] | list[str | None]],
    field: Field,
    digests: dict[str, TDigest],
) -> dict[str, list[int] | list[float | None] | list[str | None]]:
    digest = digests[field.name]
    values = observation[field.name]

    observation[field.name] = [digest.cdf(value) for value in values if value is not None]
    return observation


def collate(
    observation: dict[str, list[int] | list[float | None] | list[str | None]], params: Hyperparameters
) -> dict[str, torch.Tensor]:
    out = {}

    PAD_ = getattr(SPECIAL_TOKENS, "PAD_")
    VAL_ = getattr(SPECIAL_TOKENS, "VAL_")

    for field in params.fields:
        padding = (params.n_context - len(observation[field.name]), 0)

        if field.dtype == "continuous":
            values = np.nan_to_num(np.array(observation[field.name], dtype=float))
            indicators = np.where(np.isnan(values), PAD_, VAL_)

            out[(field.name, "values")] = pad(torch.tensor(values), pad=padding, value=0.0).float()
            out[(field.name, "lookup")] = pad(torch.tensor(indicators), pad=padding, value=PAD_)

        if field.dtype in ["discrete", "entity"]:
            lookup = np.array(observation[field.name], dtype=int) + len(SPECIAL_TOKENS)
            out[(field.name, "lookup")] = pad(torch.tensor(lookup), pad=padding, value=PAD_)

    out["label"] = torch.tensor(observation["label"])
    out["sequence_id"] = observation["sequence_id"]
    out["event_id"] = observation["event_id"]

    return out


def treeify(batch: dict[str, torch.Tensor], params: Hyperparameters) -> TensorDict:
    tree = TensorDict(batch, batch_size=params.batch_size)

    for field in params.fields:
        tree[(field.name, "dtype")] = field.dtype

    tree["label"] = batch["label"]
    tree[("meta", "sequence_id")] = json.encode(batch["sequence_id"])
    tree[("meta", "event_id")] = json.encode(batch["event_id"])

    return tree


def mask(batch: TensorDict, params: Hyperparameters) -> tuple[TensorDict, TensorDict]:
    N, L = (params.batch_size, params.n_context)

    masked = batch.clone()

    targets = {}

    is_event_masked = torch.rand(N, L).lt(params.p_mask_event)
    mask_token = torch.full((N, L), getattr(SPECIAL_TOKENS, "MASK_"))

    for field in params.fields:
        is_field_masked = torch.rand(N, L).lt(params.p_mask_field)

        for mask in [is_event_masked, is_field_masked]:
            if field.dtype == "continuous":
                masked[(field.name, "values")] *= ~mask

            if field.dtype in ["discrete", "entity", "continuous"]:
                masked[(field.name, "lookup")].masked_scatter_(mask, mask_token)

        if field.dtype == "continuous":
            targets[field.name] = (
                batch[(field.name, "values")].mul(params.n_quantiles).floor().long().add(batch[(field.name, "lookup")])
            )

        if field.dtype in ["discrete", "entity"]:
            targets[field.name] = batch[(field.name, "lookup")]

    return masked, targets


def mock(params: Hyperparameters) -> TensorDict:
    data = {}

    N = params.batch_size
    L = params.n_context

    is_padded = torch.arange(L).expand(N, L).lt(torch.randint(1, L, [N]).unsqueeze(-1)).bool()

    for field in params.fields:
        match field.dtype:
            case "continuous":
                indicators = torch.randint(0, len(SPECIAL_TOKENS), (N, L))
                indicators = torch.where(is_padded, getattr(SPECIAL_TOKENS, "PAD_"), indicators)

                is_empty = indicators.eq(getattr(SPECIAL_TOKENS, "VAL_")).long()

                data[(field.name, "lookup")] = indicators
                data[(field.name, "values")] = torch.rand(N, L).mul(is_empty)

            case "discrete":
                values = torch.randint(0, field.n_levels + len(SPECIAL_TOKENS), (N, L))
                values = torch.where(is_padded, getattr(SPECIAL_TOKENS, "PAD_"), values)
                data[(field.name, "lookup")] = values

            case "entity":
                values = torch.randint(0, L + len(SPECIAL_TOKENS), (N, L))
                values = torch.where(is_padded, getattr(SPECIAL_TOKENS, "PAD_"), values)
                data[(field.name, "lookup")] = values

        data[(field.name, "dtype")] = field.dtype

    return TensorDict(data, batch_size=N)


def stream(
    datapath: str | os.PathLike,
    mode: str,
    masks: str,
    digests: dict[str, TDigest],
    params: Hyperparameters,
) -> datapipes.iter.IterDataPipe:
    assert mode in ["pretrain", "finetune"]

    dp = (
        datapipes.iter.FileLister(datapath, masks=masks)
        .shuffle()
        .sharding_filter()
        .flatmap(read)
        .shuffle()
        .flatmap(partial(sample, params=params))
    )

    entity_hasher = partial(
        hash,
        token_map=SPECIAL_TOKENS._asdict(),
        n_context=params.n_context,
    )

    for field in params.fields:
        if field.dtype == "continuous":
            dp = dp.map(partial(cdf, field=field, digests=digests))

        if field.dtype == "entity":
            dp = dp.map(partial(entity_hasher, name=field))

    dp = (
        dp.shuffle()
        .map(partial(collate, params=params))
        .batch(params.batch_size, drop_last=True)
        .collate()
        .map(partial(treeify, params=params))
    )

    if mode == "pretrain":
        dp = dp.map(partial(mask, params=params))

    return dp


class ParquetBatchWriter(lit.callbacks.BasePredictionWriter):
    def __init__(self, outpath: str | os.PathLike):
        super().__init__("batch")
        self.outpath = outpath
        self._destination_file_exists: bool = False

    def write_on_batch_end(
        self,
        trainer: lit.Trainer,
        module: lit.LightningModule,
        prediction: torch.Tensor,
        batch_indices: Sequence[int] | None,
        batch: TensorDict,
        batch_index: int,
        dataloader_index: int,
    ) -> None:
        df = pl.DataFrame(
            prediction=prediction,
            sequence_id=batch[("meta", "sequence_id")],
            event_id=batch[("meta", "event_id")],
        ).to_arrow()

        if self._destination_file_exists:
            fastparquet.write(self.outpath, df, append=True)

        else:
            fastparquet.write(self.outpath, df)
            self._destination_file_exists = True
