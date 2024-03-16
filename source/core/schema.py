from functools import partial
from dataclasses import dataclass
from collections import namedtuple
import re

from humanize import naturalsize as bytesize

from param import Parameterized
import param
from dataclasses_json import dataclass_json

tokens = ["VAL_", "NAN_", "UNK_", "PAD_", "MASK_"]
SpecialTokens = namedtuple("SpecialTokens", tokens)
SPECIAL_TOKENS = SpecialTokens(*list(range(len(tokens))))

@dataclass_json
@dataclass
class Field:
    name: str
    dtype: str
    n_levels: None | int = None

    def __post_init__(self):
        assert re.match("^[a-z0-9_]*$", self.name), "name must only contain lowercase letters, numbers, and underscore"

        if isinstance(self.n_levels, float):
            self.n_levels = int(self.n_levels)

        assert self.dtype in ["discrete", "continuous", "entity"]

        match self.dtype:
            case "discrete":
                assert isinstance(self.n_levels, int), "n_levels must be an int for discrete fields"
                assert self.n_levels > 0, "n_levels must be greater than 0"

            case "continuous":
                assert self.n_levels is None, "n_levels must be none for continuous fields"

            case "entity":
                assert self.n_levels is None, "n_levels must be none for entity fields"

def complexity(params: "Hyperparameters") -> int:
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


class TrainingParameters(Parameterized):

    max_epochs: int = param.Integer(4, bounds=(1, 1024))
    sample_rate: float = param.Magnitude(0.001)
    check_val_every_n_epoch: int = param.Integer(1, bounds=(1, 32))
    gradient_clip_val: float = param.Magnitude(0.05)


class Hyperparameters(Parameterized):

    dev_mode: bool = param.Boolean(False)
    fields: list[Field] = param.List(item_type=Field)
    learning_rate: float = param.Magnitude(0.0001)

    # architecture hyperparameters
    batch_size: int = param.Integer(96, bounds=(1, 2048))
    n_context: int = param.Integer(192, bounds=(1, 2048))
    d_field: int = param.Integer(64, bounds=(2, 256))
    n_heads_field_encoder: int = param.Integer(16, bounds=(1, 32))
    n_layers_field_encoder: int = param.Integer(2, bounds=(1, 32))
    n_heads_event_encoder: int = param.Integer(16, bounds=(1, 32))
    n_layers_event_encoder: int = param.Integer(8, bounds=(1, 32))
    precision: int = param.Integer(8, bounds=(2, 512))
    dropout: float = param.Magnitude(0.1)
    
    # pretraining
    n_quantiles: int = param.Integer(16, bounds=(1, 512))
    p_mask_event: float = param.Magnitude(0.1)
    p_mask_field: float = param.Magnitude(0.1)
    pretrain: TrainingParameters = param.Parameter(TrainingParameters(
        max_epochs=36,
        sample_rate=0.001,
        check_val_every_n_epoch=8,
        gradient_clip_val=0.05,
    ))

    # finetuning
    n_targets: int = param.Integer(2, bounds=(2, 2048))
    freeze_epochs: int = param.Integer(1, bounds=(1, 128))
    interpolation_rate: float = param.Magnitude(0.1)
    head_shape_log_base: int = param.Integer(4, bounds=(1, 32))
    finetune: TrainingParameters = param.Parameter(TrainingParameters(
        max_epochs=36,
        sample_rate=0.001,
        check_val_every_n_epoch=8,
        gradient_clip_val=0.05,
    ))


    @property
    def n_fields(self) -> int:

        return len(self.fields)

    def complexity(self, format: bool=False) -> str | int:

        n_bytes = complexity(self)

        if format:
            return bytesize(n_bytes)
        else:
            return n_bytes

    def values(self):

        output = self.param.values()
        
        del output['name']

        output['pretrain'] = self.finetune.param.values()
        del output['pretrain']['name']

        output['finetune'] = self.finetune.param.values()
        del output['finetune']['name']
        
        return output