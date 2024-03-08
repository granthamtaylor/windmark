from dataclasses import dataclass, field
from collections import namedtuple
import re

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


@dataclass_json
@dataclass
class Hyperparameters:
    """
    Class representing the hyperparameters for a model.

    Attributes:
        fields (list[Field]): List of fields in the model.
        batch_size (int): Batch size for training.
        n_context (int): Max sequence length.
        n_targets (int, optional): Number of targets for finetuning.
        p_mask_event (float, optional): Probability of masking an event.
        p_mask_field (float, optional): Probability of masking a field.
        d_field (int, optional): Dimension of each field representation.
        n_heads_field_encoder (int, optional): Number of heads in the field encoder.
        n_layers_field_encoder (int, optional): Number of transformer layers in the field encoder.
        n_heads_event_encoder (int, optional): Number of heads in the event encoder.
        n_layers_event_encoder (int, optional): Number of transformer layers in the event encoder.
        precision (int, optional): fourier feature encoder precision.
        dropout (float, optional): Dropout rate.
        n_quantiles (int, optional): Number of quantiles for continuous fields during pretraining.
        n_fields (int): Number of unique fields (automatically calculated).
        pretrain_lr (float, optional): Learning rate for pretraining.
        pretrain_sample_rate (float, optional): Sample rate for pretraining.
        pretrain_val_interval (int, optional): Validation interval for pretraining.
        finetune_lr (float, optional): Learning rate for finetuning.
        finetune_head_lr_ratio (float, optional): Learning rate ratio between the head and the rest of the model during finetuning.
        finetune_sample_rate (float, optional): Sample rate for finetuning.
        finetune_interpolation_rate (float, optional): Interpolation rate for finetuning.
        finetune_val_interval (int, optional): Validation interval for finetuning.
    """

    fields: list[Field]

    # architecture hyperparameters
    batch_size: int = 64
    n_context: int = 8
    n_targets: int = 2
    p_mask_event: float = 0.0
    p_mask_field: float = 0.0
    d_field: int = 12
    n_heads_field_encoder: int = 4
    n_layers_field_encoder: int = 1
    n_heads_event_encoder: int = 4
    n_layers_event_encoder: int = 4
    precision: int = 8
    dropout: float = 0.1
    n_quantiles: int = 8
    head_shape_log_base: int = 4
    n_fields: int = field(init=False)

    # pretraining hyperparameters
    pretrain_lr: float = 0.0001
    pretrain_sample_rate: float = 0.001
    pretrain_val_interval: int = 8

    # finetuning hyperparameters
    finetune_lr: float = 0.00001
    finetune_head_lr_ratio: float = 10.0
    finetune_sample_rate: float = 0.1
    finetune_interpolation_rate: float = 0.001
    finetune_val_interval: int = 2

    def __post_init__(self):
        fieldnames = set([field.name for field in self.fields])
        self.n_fields = len(fieldnames)

        for reserved in ["sequence_id", "event_id", "event", "event_ids", "size", "labels", "targets"]:
            assert reserved not in fieldnames, f"{reserved} is a reserved field name"

        assert self.n_fields == len(self.fields), "multiple fields found having the same name"
        assert self.n_fields > 1

        assert self.batch_size > 0
        assert self.n_context > 0
        assert self.n_targets > 1
        assert self.n_quantiles > 1

        assert self.d_field > 0
        assert self.n_heads_field_encoder > 0
        assert self.d_field % self.n_heads_field_encoder == 0
        assert self.n_heads_event_encoder > 0
        assert (self.d_field * self.n_context) % self.n_heads_event_encoder == 0
        assert self.n_layers_field_encoder > 0
        assert self.n_layers_event_encoder > 0
        assert 0.0 <= self.dropout < 1.0
        assert self.precision > 0
        assert self.head_shape_log_base > 0

        assert 0.0 < self.pretrain_lr < 1.0
        assert 0.0 < self.pretrain_sample_rate <= 1.0
        assert self.pretrain_val_interval > 0
        assert 0.0 <= self.p_mask_event < 1.0
        assert 0.0 <= self.p_mask_field < 1.0

        assert 0.0 < self.finetune_lr < 1.0
        assert self.finetune_head_lr_ratio >= 1.0
        assert 0.0 < self.finetune_sample_rate <= 1.0
        assert 0.0 <= self.finetune_interpolation_rate <= 1.0
        assert self.finetune_val_interval > 0
