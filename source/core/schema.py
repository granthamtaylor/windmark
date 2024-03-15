from dataclasses import dataclass
from collections import namedtuple
import re

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


class Hyperparameters(Parameterized):
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
        pretrain_lr (float, optional): Learning rate for pretraining.
        pretrain_sample_rate (float, optional): Sample rate for pretraining.
        pretrain_val_interval (int, optional): Validation interval for pretraining.
        finetune_lr (float, optional): Learning rate for finetuning.
        finetune_head_lr_ratio (float, optional): Learning rate ratio between the head and the rest of the model during finetuning.
        finetune_sample_rate (float, optional): Sample rate for finetuning.
        finetune_interpolation_rate (float, optional): Interpolation rate for finetuning.
        finetune_val_interval (int, optional): Validation interval for finetuning.
    """

    fields: list[Field] = param.List(item_type=Field)

    # architecture hyperparameters
    batch_size: int = param.Integer(64, bounds=(1, 2048))
    n_context: int = param.Integer(64, bounds=(1, 2048))
    n_targets: int = param.Integer(2, bounds=(2, 2048))
    p_mask_event: float = param.Magnitude(0.0)
    p_mask_field: float = param.Magnitude(0.0)
    d_field: int = param.Integer(64, bounds=(2, 256))
    n_heads_field_encoder: int = param.Integer(8, bounds=(1, 32))
    n_layers_field_encoder: int = param.Integer(1, bounds=(1, 32))
    n_heads_event_encoder: int = param.Integer(8, bounds=(1, 32))
    n_layers_event_encoder: int = param.Integer(4, bounds=(1, 32))
    precision: int = param.Integer(8, bounds=(2, 512))
    dropout: float = param.Magnitude(0.1)
    n_quantiles: int = param.Integer(16, bounds=(1, 512))
    head_shape_log_base: int = param.Integer(4, bounds=(1, 32))

    # pretraining hyperparameters
    pretrain_lr: float = param.Magnitude(0.0001)
    pretrain_sample_rate: float = param.Magnitude(0.01)
    pretrain_val_interval: int = param.Integer(4, bounds=(1, 32))

    # finetuning hyperparameters
    finetune_lr: float = param.Magnitude(0.00001)
    finetune_head_lr_ratio: float = param.Number(10., bounds=(0, 1000))
    finetune_sample_rate: float = param.Magnitude(1.0)
    finetune_interpolation_rate: float = param.Magnitude(0.1)
    finetune_val_interval: int = param.Integer(1, bounds=(1, 32))

    @property
    def n_fields(self) -> int:
        return len(self.fields)
