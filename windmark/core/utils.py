import math
from functools import partial

import torch
from tensordict import TensorDict
from rich.console import Console
from rich.table import Table

from windmark.core.schema import Hyperparameters, SpecialTokens, DiscreteField, ContinuousField, EntityField, Field

class LabelBalancer:

    def __init__(self, labels: list[str], counts: list[int], kappa: float):
        
        for count in counts:
            assert count > 0

        assert len(labels) == len(counts)

        size: int = len(labels)
        null: float = 1 / size

        self.labels: list[str] = labels
        self.counts: list[int] = counts
        self.total: int = sum(counts)
        self.values: list[float] = [count / self.total for count in counts]

        assert math.isclose(sum(self.values), 1.0)

        assert 0.0 <= kappa <= 1.0

        self.interpolation: list[float] = [kappa * null + (1 - kappa) * value for value in self.values]

        assert math.isclose(sum(self.interpolation), 1.0)

        ratio: list[float] = [value / interpol for interpol, value in zip(self.values, self.interpolation)]

        self.thresholds: list[float] = list(map(lambda x: x / max(ratio), ratio))
        self.weights: list[float] = list(map(lambda x: sum(ratio) / (x * size * size), self.interpolation))
        
        self.kappa: float = kappa

    def show(self):

        table = Table(title=f"LabelBalancer(kappa={self.kappa:.2%})")

        table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
        
        for label in self.labels:
            table.add_column(f'"{label}"', style="magenta")
            
        def format_percent(values: list[float]) -> list[str]:
            return list(map(lambda x: f"{x:.4%}", values))
        
        def format_numbers(values: list[float]) -> list[str]:
            return list(map(lambda x: f"{x:.4}", values))
        
        def format_integers(values: list[float]) -> list[str]:
            return list(map(lambda x: f"{x:,}", values))

        table.add_row("Label Counts", *format_integers(self.counts))
        table.add_row("Population Distribution", *format_percent(self.values))
        table.add_row("Observation Distribution", *format_percent(self.interpolation))
        table.add_row("Marginal Sample Rate", *format_percent(self.thresholds))
        table.add_row("Loss Weights", *format_numbers(self.weights))

        console = Console()
        console.print(table)


def mock(params: Hyperparameters, fields: list[Field]) -> TensorDict:
    
    data = {}

    N = params.batch_size
    L = params.n_context

    is_padded = torch.arange(L).expand(N, L).lt(torch.randint(1, L, [N]).unsqueeze(-1)).bool()

    for field in fields:
        match field.type:
            case "continuous" | "temporal":
                indicators = torch.randint(0, len(SpecialTokens), (N, L))
                padded = torch.where(is_padded, SpecialTokens.PAD, indicators)
                is_empty = padded.eq(SpecialTokens.VAL).long()
                values = torch.rand(N, L).mul(is_empty)
                data[field.name] = ContinuousField(content=values, lookup=padded, batch_size=[N])

            case "discrete":
                values = torch.randint(0, field.levels + len(SpecialTokens), (N, L))
                padded = torch.where(is_padded, SpecialTokens.PAD, values)
                data[field.name] = DiscreteField(lookup=padded, batch_size=[N])

            case "entity":
                values = torch.randint(0, L + len(SpecialTokens), (N, L))
                padded = torch.where(is_padded, SpecialTokens.PAD, values)
                data[field.name] = EntityField(lookup=padded, batch_size=[N])

    return TensorDict(data, batch_size=N)


def complexity(params: Hyperparameters) -> int:
    # as per https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
    D_FFN = 2048

    # pretty good assumption you are using FP16 or BF16
    FP_PRECISION = 16

    def _calculate_bert_memory_complexity(
        batch_size: int,
        max_seq_len: int,
        d_hidden: int,
        n_heads: int,
        n_blocks: int,
        precision: int,
        d_ffn: int,
    ) -> int:
        assert isinstance(batch_size, int), "batch_size must be an integer"
        assert batch_size > 0, "batch_size must be greater than 0"

        assert isinstance(max_seq_len, int), "max_seq_len must be an integer"
        assert max_seq_len > 0, "max_seq_len must be greater than 0"

        assert isinstance(d_hidden, int), "d_hidden must be an integer"
        assert d_hidden > 0, "d_hidden must be greater than 0"

        assert isinstance(n_heads, int), "n_heads must be an integer"
        assert n_heads > 0, "n_heads must be greater than 0"

        assert isinstance(n_blocks, int), "n_blocks must be an integer"
        assert n_blocks > 0, "n_blocks must be greater than 0"

        assert isinstance(precision, int), "precision must be an integer"
        assert precision > 0, "precision must be greater than 0"

        assert isinstance(d_ffn, int), "FFN dim must be an integer"
        assert d_ffn > 0, "FFN dim must be greater than 0"

        memory = batch_size * max_seq_len * (8 * d_hidden + d_ffn)
        memory += batch_size * n_heads * max_seq_len * max_seq_len

        # the "3" comes from forward prop, backward prop, and general model overhead
        memory *= 2.5 * n_blocks * precision

        return memory

    encoder = partial(
        _calculate_bert_memory_complexity,
        precision=FP_PRECISION,
        d_ffn=D_FFN,
    )

    field = encoder(
        batch_size=params.batch_size * params.n_context,
        max_seq_len=params.n_fields,
        d_hidden=params.d_field,
        n_blocks=params.n_layers_field_encoder,
        n_heads=params.n_heads_field_encoder,
    )

    event = encoder(
        batch_size=params.batch_size,
        max_seq_len=params.n_context,
        d_hidden=params.n_fields * params.d_field,
        n_blocks=params.n_layers_event_encoder,
        n_heads=params.n_heads_event_encoder,
    )

    return int((field + event) / 8)

class SplitManager:

    def __init__(
        self,
        train: float,
        validate: float,
        test: float,
    ):

        self.train: float = train
        self.validate: float = validate
        self.test: float = test
        
        for split in [train, validate, test]:

            assert isinstance(split, float)
            assert 0.05 < split < 1.0
        
        self.ranges: dict[str, tuple[float, float]] = dict(
            train=(0.0, train),
            validate=(train, train + validate),
            test=(train + validate, 1.0),
        )
    
        assert math.isclose(sum([train, validate, test]), 1.0)

class SequenceManager:

    def __init__(
        self,
        n_sequences: int,
        n_events: int,
        shard_size: int,
        params: Hyperparameters,
        balancer: LabelBalancer,
        split: SplitManager,
    ):
        
        self.n_sequences: int = n_sequences
        self.n_events: int = n_events
        self.shard_size: int = shard_size

        self.params: Hyperparameters = params
        self.balancer: LabelBalancer = balancer
        self.split: SplitManager = split
        
        # expected pretraining steps per epoch
        pretraining_steps = int(split.train * n_events * params.pretrain_sample_rate / params.batch_size)
        
        n_labeled_events = 0
        for label_count, label_sample_rate in zip(balancer.counts, balancer.thresholds):
            n_labeled_events += label_count * label_sample_rate

        # expected finetuning steps per epoch
        finetuning_steps = int(split.train * n_labeled_events * params.finetune_sample_rate / params.batch_size)

        inference_steps = int(split.test * n_events / params.batch_size )
        
        print(f"Expected number of pretraining steps: {pretraining_steps}")
        print(f"Expected number of finetuning steps: {finetuning_steps}")
        print(f"Expected number of inference steps: {inference_steps}")