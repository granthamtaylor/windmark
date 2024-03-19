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
from pytdigest import TDigest
import lightning.pytorch as lit
import numpy as np

from source.core.schema import (
    SPECIAL_TOKENS,
    DiscreteField,
    ContinuousField,
    EntityField,
    PretrainingData,
    FinetuningData,
    InferenceData,
    SequenceData,
    Hyperparameters,
    Field
)

from source.core.utils import LabelBalancer

def read(filename):
    with open(filename, "rb") as f:
        reader = fastavro.reader(f)
        records = [record for record in reader]

    return records


def sample(
    sequence: dict,
    params: Hyperparameters,
    fields: list[Field],
    balancer: LabelBalancer,
    mode: str,
) -> list[dict[str, int | float | None]]:

    observations: list[dict[str, int | float | None]] = []

    for event in range(sequence["size"]):
        

        if (mode == 'pretrain'):
            
            if params.pretrain.sample_rate < random.random():
                continue

            label = -1

        elif (mode == 'finetune'):

            label = sequence["target"][event]

            if (label is None) or (label == -1):
                continue

            if params.finetune.sample_rate < random.random():
                continue

            if balancer.thresholds[label] < random.random():
                continue
            
        elif (mode == 'inference'):
            label = -1
        
        window = slice(max(0, event - params.n_context), event)

        observation = dict(
            sequence_id=str(sequence["sequence_id"]),
            event_id=str(sequence["event_ids"][event]),
            label=label,
        )

        for field in fields:
            observation[field.name] = sequence[field.name][window]

        observations.append(observation)

    return observations


def hash(
    observation: dict[str, list[int] | list[float | None] | list[str | None]],
    fields: list[Field],
    params: Hyperparameters,
) -> dict[str, list[int] | list[float | None] | list[str | None]]:
    
    token_map = SPECIAL_TOKENS._asdict()
    
    for field in fields:
        
        if field.type in ['entity']:
        
            field = observation[field.name]

            unique = set(field)

            [unique.remove(token) for token in token_map.keys()]

            integers = random.sample(range(len(token_map), params.n_context + len(token_map)), len(unique))

            mapping = dict(zip(unique, integers))

            mapping.update(token_map)

            observation[field.name] = [mapping[token] for token in field]

    return observation

def cdf(
    observation: dict[str, list[int] | list[float | None] | list[str | None]],
    fields: list[Field],
    digests: dict[str, TDigest],
) -> dict[str, list[int] | np.ndarray | list[str | None]]:
    
    for field in fields:
        
        if field.type in ['continuous']:

            digest: TDigest = digests[field.name]
            array = np.array(observation[field.name], dtype=np.float64)
            observation[field.name] = digest.cdf(array)

    return observation

def collate(
    observation: dict[str, list[int] | np.ndarray | list[str | None]], params: Hyperparameters,
    fields: list[Field]
) -> dict[str, torch.Tensor]:

    output = {}

    tensorclass_map = dict(
        discrete=DiscreteField,
        continuous=ContinuousField,
        entity=EntityField,
    )

    for field in fields:
        values = observation[field.name]
        tensorclass = tensorclass_map[field.type]
        output[field.name] = tensorclass.collate(values, params=params)

    inputs = TensorDict(output, batch_size=1)
    labels = torch.tensor(observation["label"])

    return inputs, labels

def stack(batch: list[tuple[TensorDict, torch.Tensor]]):
    
    # FIXME something is broken here

    inputs, targets = zip(*batch)
    
    inputs = torch.stack(inputs, dim=0).squeeze(dim=1).auto_batch_size_(batch_dims=1)

    targets = torch.stack(targets, dim=0)
    
    return inputs, targets

def mask(
    batch: TensorDict,
    params: Hyperparameters,
    fields: list[Field],
) -> tuple[TensorDict, TensorDict]:
    
    inputs, _ = batch
    
    N, L = (params.batch_size, params.n_context)
    
    targets = {}

    is_event_masked = torch.rand(N, L).lt(params.p_mask_event)

    for field in fields:

        targets[field.name] = inputs[field.name].target(params=params)
        inputs[field.name].mask(is_event_masked, params=params)
        
    targets = TensorDict(targets, batch_size=params.batch_size)

    return inputs, targets

def to_tensorclass(
    batch: tuple[TensorDict, torch.Tensor],
    params: Hyperparameters,
    fields: list[Field],
    mode: str,
) -> SequenceData:
    
    if mode == 'pretrain':
        batch: tuple[TensorDict, TensorDict] = mask(
            batch=batch,
            params=params,
            fields=fields
        )
    
    tensorclass_map = dict(
        pretrain=PretrainingData,
        finetune=FinetuningData,
        inference=InferenceData,
    )

    tensorclass = tensorclass_map[mode]
    
    return tensorclass.from_stream(batch, batch_size=params.batch_size)

def stream(
    datapath: str | os.PathLike,
    mode: str,
    masks: str,
    centroids: dict[str, np.ndarray],
    fields: list[Field],
    params: Hyperparameters,
    balancer: LabelBalancer,
) -> datapipes.iter.IterDataPipe:
    
    digests = {field: TDigest.of_centroids(centroid) for field, centroid in centroids.items()}

    assert mode in ["pretrain", "finetune", "inference"]
    
    print(f"creating {mode} datapipe")
    
    sampler = partial(sample, fields=fields, params=params, balancer=balancer, mode=mode)

    return (
        datapipes.iter.FileLister(datapath, masks=masks)
        .shuffle()
        .sharding_filter()
        .flatmap(read)
        .shuffle()
        .flatmap(sampler)
        .map(partial(cdf, fields=fields, digests=digests))
        .map(partial(hash, fields=fields, params=params))
        .shuffle()
        .map(partial(collate, fields=fields, params=params))
        .batch(params.batch_size, drop_last=True)
        .map(stack)
        .map(partial(to_tensorclass, params=params, fields=fields, mode=mode))
    )


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
        batch: SequenceData,
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
