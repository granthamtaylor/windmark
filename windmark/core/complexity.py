from functools import partial

from windmark.core.structs import Hyperparameters
from windmark.core.managers import SystemManager


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
    memory *= 3 * n_blocks * precision

    return memory


def complexity(params: Hyperparameters, manager: SystemManager) -> int:
    # as per https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
    D_FFN = 2048

    # pretty good assumption you are using FP16 or BF16
    FP_PRECISION = 16

    encoder = partial(
        _calculate_bert_memory_complexity,
        precision=FP_PRECISION,
        d_ffn=D_FFN,
    )

    field = encoder(
        batch_size=params.batch_size * params.n_context,
        max_seq_len=len(manager.schema),
        d_hidden=params.d_field,
        n_blocks=params.n_layers_field_encoder,
        n_heads=params.n_heads_field_encoder,
    )

    event = encoder(
        batch_size=params.batch_size,
        max_seq_len=params.n_context,
        d_hidden=len(manager.schema) * params.d_field,
        n_blocks=params.n_layers_event_encoder,
        n_heads=params.n_heads_event_encoder,
    )

    return int((field + event) / 8)
