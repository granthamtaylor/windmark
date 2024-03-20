import math
import os
from collections import OrderedDict
from functools import partial, partialmethod

import lightning.pytorch as lit
import torch
import torchmetrics
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from lightning.fabric.utilities.throughput import measure_flops
from lightning.pytorch.trainer.states import TrainerFn as StageName
from tensordict import TensorDict
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from torchdata import datapipes
import numpy as np
import humanize

from source.core.iterops import stream
from source.core.schema import SPECIAL_TOKENS, Field, ContinuousField, DiscreteField, EntityField, SequenceData, Hyperparameters
from source.core.utils import LabelBalancer, mock, complexity

@jaxtyped(typechecker=beartype)
def _squarify(tensor: Tensor) -> Tensor:
    return ~(tensor.unsqueeze(-1) & tensor.unsqueeze(-2))


@jaxtyped(typechecker=beartype)
def create_attention_masks(inputs: TensorDict, fields: list[Field]) -> tuple[Tensor, Tensor]:
    is_null = []

    for field in fields:
        values = inputs[(field.name, "lookup")]

        is_padded = values.eq(getattr(SPECIAL_TOKENS, "PAD_"))
        is_nan = values.eq(getattr(SPECIAL_TOKENS, "NAN_"))
        is_unknown = values.eq(getattr(SPECIAL_TOKENS, "UNK_"))

        is_null.append(is_padded | is_nan | is_unknown)

    is_null = torch.stack(is_null, dim=-1)

    N, L, F = is_null.shape

    field_mask = ~is_null.view(N * L, F)
    event_mask = ~is_null.amin(-1)

    return (_squarify(field_mask), _squarify(event_mask))


class LearnedTensor(torch.nn.Module):
    """
    LearnedTensor is a PyTorch module that initializes a tensor with normal distribution and learns its values during training.
    """

    def __init__(self, *sizes: int):
        """
        Initializes the LearnedTensor.

        Args:
            sizes (int): The sizes of the dimensions of the tensor.
        """

        super().__init__()

        for dim in sizes:
            assert isinstance(dim, int), "each dim must be of type int"
            assert dim >= 1, "each dim must be greater than 0"

        tensor = torch.normal(mean=0.0, std=1e-4, size=tuple(sizes))

        self.tensor = torch.nn.Parameter(tensor)

    @jaxtyped(typechecker=beartype)
    def forward(self) -> Float[Tensor, "..."]:
        """
        Returns the learned tensor.

        Returns:
            Float[Tensor, '...']: The learned tensor.
        """

        return self.tensor


class DiscreteFieldEmbedder(torch.nn.Module):
    def __init__(self, params: Hyperparameters, field: Field):
        """
        Initialize discrete field embedder.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
        """
        super().__init__()

        self.field: Field = field
        self.embeddings = torch.nn.Embedding(field.levels + len(SPECIAL_TOKENS), params.d_field)

    def forward(self, inputs: DiscreteField) -> Tensor:
        return self.embeddings(inputs.lookup)


class EntityFieldEmbedder(torch.nn.Module):
    def __init__(self, params: Hyperparameters, field: Field):
        super().__init__()
        """
        Initialize entity field embedder.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
        """

        self.field: Field = field
        self.embeddings = torch.nn.Embedding(params.n_context + len(SPECIAL_TOKENS), params.d_field)

    def forward(self, inputs: EntityField) -> Tensor:
        return self.embeddings(inputs.lookup)


class ContinuousFieldEmbedder(torch.nn.Module):
    """
    ContinuousFieldEmbedder is a PyTorch module that encodes features using Fourier features.

    Attributes:
        linear (torch.nn.Linear): A linear layer for transforming the input.
        positional (torch.nn.Embedding): An embedding layer for positional encoding.
        weights (Tensor): The weights for the Fourier features.
    """

    def __init__(self, params: Hyperparameters, field: Field):
        """
        Initialize continuous field embedder.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
        """

        super().__init__()

        self.field: Field = field
        self.linear = torch.nn.Linear(2 * params.precision, params.d_field)
        self.positional = torch.nn.Embedding(len(SPECIAL_TOKENS), params.d_field)

        weights = torch.logspace(-params.precision, 1, params.precision, base=2).mul(math.pi).unsqueeze(dim=0)

        self.register_buffer("weights", weights)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        inputs: ContinuousField,
    ) -> Float[Tensor, "N L F"]:
        """
        Performs the forward pass of the FourierFeatureEncoder.

        Args:
            inputs (Float[Tensor, "N L"]): The input tensor.

        Returns:
            Float[Tensor, "N L F"]: The Fourier features of the input.
        """

        values = inputs.content
        indicators = inputs.lookup

        assert values.shape == indicators.shape, "values and indicators must always have the same shape"

        assert torch.all(values.mul(indicators).eq(0.0)), "values should be imputed if not null, padded, or masked"

        assert torch.all(values.lt(1.0)), "values should be less than 1.0"

        assert torch.all(values.ge(0.0)), "values should be greater than or equal to 0.0"

        N, L = values.shape

        # weight inputs with buffers of precision bands
        weighted = values.view(N * L).unsqueeze(dim=1).mul(self.weights)

        # apply sine and cosine functions to weighted inputs
        fourier = torch.sin(weighted), torch.cos(weighted)

        # project sinusoidal representations with MLP
        projections = self.linear(torch.cat(fourier, dim=1)).view(N, L, -1)

        # embed null indicators
        positional = self.positional(indicators)

        return projections + positional


class ModularAttributeEmbeddingSystem(torch.nn.Module):
    """
    ModularAttributeEmbeddingSystem is a PyTorch module for embedding fields.
    """

    def __init__(self, params: Hyperparameters, fields: list[Field]):
        """
        Initialize moduler field embedder.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
        """
        super().__init__()

        embedders: dict[str, torch.nn.Module] = {}

        embedder_map: dict[str, type] = dict(
            discrete=DiscreteFieldEmbedder,
            continuous=ContinuousFieldEmbedder,
            entity=EntityFieldEmbedder,
        )

        for field in fields:
            embedder = embedder_map[field.type]
            embedders[field.name] = embedder(params=params, field=field)

        self.embedders = torch.nn.ModuleDict(embedders)

    @jaxtyped(typechecker=beartype)
    def forward(self, inputs: TensorDict) -> Float[Tensor, "N L F C"]:
        """
        Performs the forward pass of the ModularAttributeEmbeddingSystem.

        Args:
            inputs (TensorDict[Float[Tensor, "N L"] | Int[Tensor, "N L"]]): The input tensor.

        Returns:
            Float[Tensor, "N L F"]: The embedded fields.
        """

        embeddings = []

        for field in inputs.keys():
            if field in self.embedders.keys():
                embedding = self.embedders[field](inputs[field])
                embeddings.append(embedding)

        # N L C F
        stacked = torch.stack(embeddings, dim=-1)

        # N L F C
        return stacked.permute(0, 1, 3, 2)


class FieldEncoder(torch.nn.Module):
    """
    FieldEncoder is a PyTorch module for encoding fields.
    """

    def __init__(self, params: Hyperparameters):
        """
        Initialize field transformer encoder.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
        """

        super().__init__()
        
        
        identity = torch.eye(params.n_fields).bool().unsqueeze(0)
        self.register_buffer('identity', identity)

        self.H = params.n_heads_field_encoder

        layer = torch.nn.TransformerEncoderLayer(
            d_model=params.d_field,
            nhead=params.n_heads_field_encoder,
            batch_first=True,
            dropout=params.dropout,
        )

        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=params.n_layers_field_encoder)

        self.positional = LearnedTensor(params.n_fields, params.d_field)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        fields: Float[Tensor, "N L F C"],
        mask: Bool[Tensor, "NL F F"],
    ) -> Float[Tensor, "N L FC"]:
        """
        Performs the forward pass of the FieldEncoder.

        Args:
            inputs (Float[Tensor, "N L F C"]): The input tensor.

        Returns:
            Float[Tensor, "N L F C"]: The encoded fields.
        """

        N, L, F, C = fields.shape
        H = self.H

        # NL F C
        batched = fields.view(N * L, F, C)

        # NL F C
        batched += self.positional().unsqueeze(dim=0).expand(N * L, F, C)

        # NLH F F
        identity = self.identity.expand((N * L * H, F, F))

        # NLH F F
        masks = torch.repeat_interleave(mask, H, dim=0) & ~identity

        # NL F C
        events = self.encoder(batched, mask=masks).view(N, L, F * C)

        # N L FC
        return events


class EventEncoder(torch.nn.Module):
    """
    EventEncoder is a PyTorch module for encoding events.
    """

    def __init__(self, params: Hyperparameters):
        """
        Initialize the event transformer encoder.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
        """

        super().__init__()

        self.H = params.n_heads_event_encoder

        layer = torch.nn.TransformerEncoderLayer(
            d_model=params.n_fields * params.d_field,
            nhead=params.n_heads_event_encoder,
            batch_first=True,
            dropout=params.dropout,
        )

        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=params.n_layers_event_encoder)

        self.positional = LearnedTensor(params.n_context, params.n_fields * params.d_field)
        self.class_token = LearnedTensor(1, params.n_fields * params.d_field)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        events: Float[Tensor, "N L FC"],
        mask: Bool[Tensor, "N L L"],
    ) -> tuple[
        Float[Tensor, "N FC"],
        Float[Tensor, "N L FC"],
    ]:
        N, L, FC = events.shape
        H = self.H

        # N L FC
        events += self.positional().unsqueeze(0).expand(N, L, FC)

        # N 1 FC
        class_token = self.class_token().unsqueeze(0).expand(N, 1, FC)

        # N L+1 FC
        concatenated = torch.cat((class_token, events), dim=1)

        # NH L+1 L+1
        masks = torch.nn.functional.pad(mask, (1, 0, 1, 0), value=0).repeat_interleave(H, dim=0)

        # N L+1 FC
        encoded = self.encoder(concatenated, mask=masks)

        # (N 1 FC), (N L FC)
        sequence, events = encoded.split((1, L), dim=1)

        return sequence.squeeze(dim=1), events


class EventDecoder(torch.nn.Module):
    """
    EventDecoder is a PyTorch module for decoding masked events from their contextualized representations.
    """

    def __init__(self, params: Hyperparameters, fields: list[Field]):
        """
        Initialize the event decoder.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
        """

        super().__init__()

        projections = {}

        for field in fields:
            match field.type:
                case "discrete":
                    d_target = field.levels

                case "entity":
                    d_target = params.n_context

                case "continuous":
                    d_target = params.n_quantiles

            projections[field.name] = torch.nn.Conv1d(
                in_channels=params.n_fields * params.d_field,
                out_channels=d_target + len(SPECIAL_TOKENS),
                kernel_size=1,
            )

        self.projections = torch.nn.ModuleDict(projections)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        inputs: Float[Tensor, "N L FC"],
    ) -> TensorDict[Float[Tensor, "N L ?"]]:
        """
        Performs the forward pass of the EventDecoder.

        Args:
            inputs (Float[Tensor, "N L FC"]): The input tensor.

        Returns:
            TensorDict[Float[Tensor, "N L ?"]]: A dictionary of output tensors for each field.
        """

        N = inputs.shape[0]

        # N, FC, L
        permuted = inputs.permute(0, 2, 1)

        events = {}

        for field, projection in self.projections.items():
            events[field] = projection(permuted)

        # N, L, ?
        return TensorDict(events, batch_size=N)


class DecisionHead(torch.nn.Module):
    """
    A PyTorch module representing a classification head. This module is used to map the encoded fields to the target classes.
    """

    def __init__(self, params: Hyperparameters):
        """
        Initialize classification decision head.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
        """
        super().__init__()

        hidden = list(
            map(
                lambda x: params.head_shape_log_base**x,
                range(
                    math.ceil(math.log(params.n_targets, params.head_shape_log_base)),
                    math.ceil(math.log(params.n_fields * params.d_field, params.head_shape_log_base)),
                ),
            )
        )

        hidden.reverse()

        dims = [params.n_fields * params.d_field, *hidden, params.n_targets]

        layers = []

        for index, sizes in enumerate(zip(dims[:-1], dims[1:])):
            if index < len(dims) - 2:
                activation = torch.nn.ReLU()
            else:
                activation = torch.nn.Identity()

            layers.append(
                torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("dense", torch.nn.Linear(*sizes)),
                            ("act", activation),
                        ]
                    )
                )
            )

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, inputs: Float[Tensor, "N FC"]) -> Float[Tensor, "N T"]:
        """
        Defines the forward pass of the ClassificationHead.

        Args:
            inputs (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the MLP.
        """

        return self.mlp(inputs)


def create_metrics(params: Hyperparameters) -> torch.nn.ModuleDict:
    metrics = dict(
        ap=torchmetrics.AveragePrecision,
        f1=torchmetrics.F1Score,
        auc=torchmetrics.AUROC,
        acc=torchmetrics.Accuracy,
    )

    stratas = torch.nn.ModuleDict(
        {
            "train_metrics": torch.nn.ModuleDict(),
            "validate_metrics": torch.nn.ModuleDict(),
            "test_metrics": torch.nn.ModuleDict(),
        }
    )

    config = dict(task="multiclass", num_classes=params.n_targets)

    for strata in stratas.keys():
        for name, Metric in metrics.items():
            stratas[strata][name] = Metric(**config)

    return stratas


def step(
    self: "SequenceModule",
    batch: SequenceData,
    strata: str,
) -> Tensor:
    assert strata in ["train", "validate", "test", "predict"]

    predictions, reconstruction = self.forward(batch.inputs)

    log = partial(
        self.log,
        on_step=False,
        on_epoch=True,
        batch_size=self.params.batch_size,
    )

    if self._mode == "pretrain":

        losses = []

        for field in self.fields:
            
            values = reconstruction[field.name]
            targets = batch.targets[field.name]
            loss = cross_entropy(values, targets.lookup, reduction='none').mul(targets.is_masked).mean()
            losses.append(loss)
            log(f"{self._mode}-{strata}/{field.name}-loss", loss)

        total_loss = torch.stack(losses).sum()
        log(f"{self._mode}-{strata}/loss", total_loss, prog_bar=True)
        return total_loss

    elif self._mode == "finetune":

        loss = cross_entropy(predictions, batch.targets, weight=self.weights)
        log(f"{self._mode}-{strata}/loss", loss, prog_bar=True)

        logits = torch.nn.functional.softmax(predictions, dim=1)
        for title, metric in self.metrics[f"{strata}_metrics"].items():
            metric(logits, batch.targets)
            log(f"{self._mode}-{strata}/{title}", metric)

        return loss

    elif self._mode == "inference":
        return predictions


def dataloader(self: "SequenceModule", strata: str) -> DataLoader:
    assert strata in ["train", "validate", "test", "predict"]

    print(f"creating {self._mode} dataloader for strata {strata}")

    pipe = self.pipes[strata]
    loader = DataLoader(
        pipe, 
        batch_size=None, 
        num_workers=15, 
        collate_fn=lambda x: x,
        pin_memory=False,
    )
    self.dataloaders[strata] = loader

    return loader


class SequenceModule(lit.LightningModule):
    def __init__(
        self,
        datapath: str | os.PathLike,
        params: Hyperparameters,
        fields: list[Field],
        centroids: dict[str, np.ndarray],
        balancer: LabelBalancer,
    ):
        super().__init__()

        assert os.path.exists(datapath)

        assert isinstance(params, Hyperparameters)

        for field in fields:
            assert field.is_valid, f'field {field.name} is not valid'

        self.datapath: str | os.PathLike = datapath

        self.params: Hyperparameters = params
        self.fields: list[Field] = fields
        self.save_hyperparameters(params.values())

        self.lr = params.learning_rate

        self.modular_field_embedder = ModularAttributeEmbeddingSystem(params=params, fields=fields)
        self.field_encoder = FieldEncoder(params=params)
        self.event_encoder = EventEncoder(params=params)
        self.event_decoder = EventDecoder(params=params, fields=fields)
        self.decision_head = DecisionHead(params=params)

        self.metrics = create_metrics(params=params)

        self._mode: str = "pretrain"
        self.balancer = balancer

        self.register_buffer('weights', torch.tensor(self.balancer.weights))

        self.dataloaders: dict[str, DataLoader] = {}
        self.pipes: dict[str, datapipes.iter.IterDataPipe] = {}
        self.centroids: dict[str, np.ndarray] = centroids

        self.example_input_array = sample = mock(params=params, fields=fields)

        self.flops_per_batch = measure_flops(self, lambda: self.forward(sample))


    @jaxtyped(typechecker=beartype)
    def forward(self, inputs: TensorDict) -> tuple[Float[Tensor, "N T"], TensorDict]:
        """
        Performs the forward pass of the encoder.

        Args:
            events (Float[Tensor, "N L FC"]): The input tensor.

        Returns:
            tuple[Tensor, TensorDict]:
            A tuple containing two tensors. The first tensor represents the supervised classification predictions and the second tensor represents the decoded, reconstructed events.
        """

        field_mask, event_mask = create_attention_masks(inputs, fields=self.fields)

        fields = self.modular_field_embedder(inputs)
        events = self.field_encoder(fields, mask=field_mask)
        sequence, representations = self.event_encoder(events, mask=event_mask)

        predictions = self.decision_head(sequence)
        reconstructions = self.event_decoder(representations)

        return predictions, reconstructions

    def configure_optimizers(self) -> torch.optim.AdamW:

        print(f"configuring optimizer for mode {self._mode}")

        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.params.weight_decay
        )

    training_step = partialmethod(step, strata="train")
    validation_step = partialmethod(step, strata="validate")
    test_step = partialmethod(step, strata="test")
    predict_step = partialmethod(step, strata="predict")

    def setup(self, stage: StageName):
        pipe = partial(
            stream,
            datapath=self.datapath,
            mode=self._mode,
            centroids=self.centroids,
            params=self.params,
            balancer=self.balancer,
            fields=self.fields,
        )

        assert stage in ["fit", "validate", "test", "predict"]

        print(f"setting up {self._mode} for stage {stage.value}")

        mapping = dict(
            fit=["train", "validate"],
            validate=["validate"],
            test=["test"],
            predict=["predict"],
        )


        for strata in mapping[stage]:            
            dataset = (strata if strata != "predict" else "test")
            self.pipes[strata] = pipe(masks=f"{dataset}-*.avro")

    def teardown(self, stage: StageName):
        assert stage in ["fit", "validate", "test", "predict"]
        print(f"tearing down {self._mode} for stage {stage.value}")

        mapping = dict(
            fit=["train", "validate"],
            validate=["validate"],
            test=["test"],
            predict=["predict"],
        )

        for strata in mapping[stage]:

            del self.pipes[strata]
            del self.dataloaders[strata]

    train_dataloader = partialmethod(dataloader, strata="train")
    val_dataloader = partialmethod(dataloader, strata="validate")
    test_dataloader = partialmethod(dataloader, strata="test")
    predict_dataloader = partialmethod(dataloader, strata="predict")

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode: str):
        assert mode in ["pretrain", "finetune", "inference"]

        print(f"setting mode to {mode}")

        self._mode = mode

    def show(self):

        memory = humanize.naturalsize(complexity(self.params))
        flops = humanize.metric(self.flops_per_batch, "V")
