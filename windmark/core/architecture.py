import math
import os
from collections import OrderedDict
from functools import partial, partialmethod
from typing import Annotated

import lightning.pytorch as lit
import torch
import torchmetrics
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from lightning.fabric.utilities.throughput import measure_flops
from tensordict import TensorDict
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from torchdata import datapipes

from windmark.core.managers import SystemManager
from windmark.core.operators import collate, stream, mock
from windmark.core.constructs import (
    ContinuousField,
    DiscreteField,
    EntityField,
    TemporalField,
    TargetField,
    PretrainingData,
    SupervisedData,
    OutputData,
    Field,
    Hyperparameters,
    SequenceData,
    Tokens,
)


@jaxtyped(typechecker=beartype)
def create_attention_masks(inputs: TensorDict, manager: SystemManager) -> tuple[Tensor, Tensor]:
    is_non_valued = []

    for field in manager.schema.fields:
        values = inputs[(field.name, "lookup")]

        is_padded = values.eq(Tokens.PAD)
        is_unknown = values.eq(Tokens.UNK)
        is_non_valued.append(is_padded | is_unknown)

    is_non_valued = torch.stack(is_non_valued, dim=-1)

    N, L, F = is_non_valued.shape

    field_mask = ~is_non_valued.view(N * L, F)
    event_mask = ~is_non_valued.amin(-1)

    # FIXME this could be better
    return (
        ~(field_mask.unsqueeze(-1) & field_mask.unsqueeze(-2)),
        ~(event_mask.unsqueeze(-1) & event_mask.unsqueeze(-2)),
    )


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
    def __init__(self, params: Hyperparameters, manager: SystemManager, field: Field):
        """
        Initialize discrete field embedder.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
            manager (SystemManager): The pipeline system manager.
            field (Field): The field to be embedded
        """
        super().__init__()

        self.field: Field = field
        self.embeddings = torch.nn.Embedding(manager.levelsets.get_size(field) + len(Tokens), params.d_field)

    def forward(self, inputs: DiscreteField) -> Tensor:
        return self.embeddings(inputs.lookup)


class EntityFieldEmbedder(torch.nn.Module):
    def __init__(self, params: Hyperparameters, manager: SystemManager, field: Field):
        super().__init__()
        """
        Initialize entity field embedder.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
            manager (SystemManager): The pipeline system manager.
            field (Field): The field to be embedded
        """

        self.field: Field = field
        self.embeddings = torch.nn.Embedding(params.n_context + len(Tokens), params.d_field)

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

    def __init__(self, params: Hyperparameters, manager: SystemManager, field: Field):
        """
        Initialize continuous field embedder.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
            manager (SystemManager): The pipeline system manager.
            field (Field): The field to be embedded
        """

        super().__init__()

        self.field: Field = field

        offset = 3

        weights = torch.logspace(start=-params.n_bands, end=offset, steps=params.n_bands + offset + 1, base=2)

        self.linear = torch.nn.Linear(2 * len(weights), params.d_field)
        self.register_buffer("weights", weights.mul(math.pi).unsqueeze(dim=0))

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        inputs: ContinuousField | TemporalField,
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

        assert torch.all(
            values.mul(indicators).eq(0.0), dim=None
        ), "values should be imputed if not null, padded, or masked"

        assert torch.all(values.lt(1.0), dim=None), "values should be less than 1.0"

        assert torch.all(values.ge(0.0), dim=None), "values should be greater than or equal to 0.0"

        N, L = values.shape

        # weight inputs with buffers of precision bands
        weighted = values.sub(indicators).view(N * L).unsqueeze(dim=1).mul(self.weights)

        # apply sine and cosine functions to weighted inputs
        fourier = torch.sin(weighted), torch.cos(weighted)

        # project sinusoidal representations with MLP
        projections = self.linear(torch.cat(fourier, dim=1)).view(N, L, -1)

        return projections


class TemporalFieldEmbedder(torch.nn.Module):
    def __init__(self, params: Hyperparameters, manager: SystemManager, field: Field):
        """
        Initialize continuous field embedder.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
            manager (SystemManager): The pipeline system manager.
            field (Field): The field to be embedded
        """

        super().__init__()

        self.field: Field = field

        offset = 0

        weights = torch.logspace(start=-params.n_bands, end=offset, steps=params.n_bands + offset + 1, base=2)

        self.linear = torch.nn.Linear(2 * len(weights), params.d_field)
        self.register_buffer("weights", weights.mul(math.pi).unsqueeze(dim=0))

        self.embeddings = torch.nn.ModuleDict(
            dict(
                week_of_year=torch.nn.Embedding(53 + len(Tokens), params.d_field),
                day_of_week=torch.nn.Embedding(7 + len(Tokens), params.d_field),
            )
        )

    @jaxtyped(typechecker=beartype)
    def forward(self, inputs: TemporalField) -> Float[Tensor, "N L F"]:
        """
        Performs the forward pass of the FourierFeatureEncoder.

        Args:
            inputs (Float[Tensor, "N L"]): The input tensor.

        Returns:
            Float[Tensor, "N L F"]: The Fourier features of the input.
        """

        assert inputs.time_of_day.shape == inputs.lookup.shape, "values and indicators must always have the same shape"

        assert torch.all(
            inputs.time_of_day.mul(inputs.lookup).eq(0.0), dim=None
        ), "values should be imputed if not null, padded, or masked"

        assert torch.all(inputs.time_of_day.lt(1.0), dim=None), "values should be less than 1.0"

        assert torch.all(inputs.time_of_day.ge(0.0), dim=None), "values should be greater than or equal to 0.0"

        N, L = inputs.time_of_day.shape

        # weight inputs with buffers of precision bands
        weighted = inputs.time_of_day.view(N * L).unsqueeze(dim=1).mul(self.weights)

        # apply sine and cosine functions to weighted inputs
        fourier = torch.sin(weighted), torch.cos(weighted)

        # project sinusoidal representations with MLP
        projections = self.linear(torch.cat(fourier, dim=1)).view(N, L, -1)

        projections += self.embeddings["week_of_year"](inputs.week_of_year)
        projections += self.embeddings["day_of_week"](inputs.day_of_week)

        return projections


class ModularFieldEmbeddingSystem(torch.nn.Module):
    """
    ModularFieldEmbeddingSystem is a PyTorch module for embedding fields.
    """

    def __init__(self, params: Hyperparameters, manager: SystemManager):
        """
        Initialize modular field embedder.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
            manager (SystemManager): The pipeline system manager.
        """
        super().__init__()

        embedders: dict[str, torch.nn.Module] = {}

        embedder_map: dict[str, type] = dict(
            discrete=DiscreteFieldEmbedder,
            continuous=ContinuousFieldEmbedder,
            entity=EntityFieldEmbedder,
            temporal=TemporalFieldEmbedder,
        )

        for field in manager.schema.fields:
            embedder = embedder_map[field.type]
            embedders[field.name] = embedder(params=params, manager=manager, field=field)

        self.embedders = torch.nn.ModuleDict(embedders)

    @jaxtyped(typechecker=beartype)
    def forward(self, inputs: TensorDict) -> Float[Tensor, "N L F C"]:
        """
        Performs the forward pass of the ModularFieldEmbeddingSystem.

        Args:
            inputs (TensorDict[Float[Tensor, "N L"] | Int[Tensor, "N L"]]): The input tensor.

        Returns:
            Float[Tensor, "N L F"]: The embedded fields.
        """

        embeddings = []

        for field in self.embedders.keys():
            embedder = self.embedders[field]
            embedding = embedder(inputs[field])
            embeddings.append(embedding)

        # N L C F
        stacked = torch.stack(embeddings, dim=-1)

        # N L F C
        return stacked.permute(0, 1, 3, 2)


class FieldEncoder(torch.nn.Module):
    """
    FieldEncoder is a PyTorch module for encoding fields.
    """

    def __init__(self, params: Hyperparameters, manager: SystemManager):
        """
        Initialize field transformer encoder.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
            manager (SystemManager): The pipeline system manager.
        """

        super().__init__()

        identity = torch.eye(len(manager.schema)).bool().unsqueeze(0)
        self.register_buffer("identity", identity)

        self.H = params.n_heads_field_encoder

        layer = torch.nn.TransformerEncoderLayer(
            d_model=params.d_field,
            nhead=params.n_heads_field_encoder,
            batch_first=True,
            dropout=params.dropout,
        )

        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=params.n_layers_field_encoder)

        self.positional = LearnedTensor(len(manager.schema), params.d_field)

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
        # H = self.H

        # NL F C
        batched = fields.view(N * L, F, C)

        # NL F C
        batched += self.positional().unsqueeze(dim=0).expand(N * L, F, C)

        # NLH F F
        # identity = self.identity.expand((N * L * H, F, F))

        # NLH F F
        # masks = torch.repeat_interleave(mask, H, dim=0) & ~identity

        # NL F C
        # events = self.encoder(batched, mask=masks).view(N, L, F * C)
        events = self.encoder(batched).view(N, L, F * C)

        # N L FC
        return events


class EventEncoder(torch.nn.Module):
    """
    EventEncoder is a PyTorch module for encoding events.
    """

    def __init__(self, params: Hyperparameters, manager: SystemManager):
        """
        Initialize the event transformer encoder.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
            manager (SystemManager): The pipeline system manager.
        """

        super().__init__()

        self.H = params.n_heads_event_encoder

        layer = torch.nn.TransformerEncoderLayer(
            d_model=len(manager.schema) * params.d_field,
            nhead=params.n_heads_event_encoder,
            batch_first=True,
            dropout=params.dropout,
        )

        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=params.n_layers_event_encoder)

        self.positional = LearnedTensor(params.n_context, len(manager.schema) * params.d_field)
        self.class_token = LearnedTensor(1, len(manager.schema) * params.d_field)

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
        # H = self.H

        # N L FC
        events += self.positional().unsqueeze(0).expand(N, L, FC)

        # N 1 FC
        class_token = self.class_token().unsqueeze(0).expand(N, 1, FC)

        # N L+1 FC
        concatenated = torch.cat((class_token, events), dim=1)

        # NH L+1 L+1
        # masks = torch.nn.functional.pad(mask, (1, 0, 1, 0), value=0).repeat_interleave(H, dim=0)

        # N L+1 FC
        # encoded = self.encoder(concatenated, mask=masks)
        encoded = self.encoder(concatenated)

        # (N 1 FC), (N L FC)
        sequence, events = encoded.split((1, L), dim=1)

        return sequence.squeeze(dim=1), events


class EventDecoder(torch.nn.Module):
    """
    EventDecoder is a PyTorch module for decoding masked events from their contextualized representations.
    """

    def __init__(self, params: Hyperparameters, manager: SystemManager):
        """
        Initialize the event decoder.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
            manager (SystemManager): The pipeline system manager.
        """

        super().__init__()

        projections = {}

        for field in manager.schema.fields:
            match field.type:
                case "discrete":
                    d_target = manager.levelsets.get_size(field)

                case "entity":
                    d_target = params.n_context

                case "continuous":
                    d_target = params.n_quantiles

                case "temporal":
                    d_target = (366 * 24) + len(Tokens)

                case _:
                    raise NotImplementedError

            projections[field.name] = torch.nn.Conv1d(
                in_channels=len(manager.schema) * params.d_field,
                out_channels=d_target + len(Tokens),
                kernel_size=1,
            )

        self.projections = torch.nn.ModuleDict(projections)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        inputs: Float[Tensor, "N L FC"],
    ) -> Annotated[TensorDict, Float[Tensor, "N L _"]]:
        """
        Performs the forward pass of the EventDecoder.

        Args:
            inputs (Float[Tensor, "N L FC"]): The input tensor.

        Returns:
            TensorDict[Float[Tensor, "N L _"]]: A dictionary of output tensors for each field.
        """

        N = inputs.shape[0]

        # N, FC, L
        permuted = inputs.permute(0, 2, 1)

        events = {}

        for field, projection in self.projections.items():
            events[field] = projection(permuted).permute(0, 2, 1)

        # N, L, ?
        return TensorDict(events, batch_size=N)


class DecisionHead(torch.nn.Module):
    """
    A PyTorch module representing a classification head. This module is used to map the encoded fields to the target classes.
    """

    def __init__(self, params: Hyperparameters, manager: SystemManager):
        """
        Initialize classification decision head.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
            manager (SystemManager): The pipeline system manager.
        """
        super().__init__()

        hidden = list(
            map(
                lambda x: params.head_shape_log_base**x,
                range(
                    math.ceil(math.log(manager.task.n_targets, params.head_shape_log_base)),
                    math.ceil(math.log(len(manager.schema) * params.d_field, params.head_shape_log_base)),
                ),
            )
        )

        hidden.reverse()

        dims = [len(manager.schema) * params.d_field, *hidden, manager.task.n_targets]

        layers = []

        norm = torch.nn.utils.parametrizations.weight_norm

        for index, sizes in enumerate(zip(dims[:-1], dims[1:])):
            if index < len(dims) - 2:
                activation = torch.nn.LeakyReLU()
            else:
                activation = torch.nn.Identity()

            layers.append(
                torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("dense", norm(torch.nn.Linear(*sizes))),
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


def create_metrics(manager: SystemManager) -> torch.nn.ModuleDict:
    """Create supervised module metrics

    Args:
        manager (SystemManager): The pipeline system manager.

    Returns:
        torch.nn.ModuleDict: Nested dictionaries of metrics (strata > metrics > instances)
    """

    n_targets = manager.task.n_targets

    metrics = dict(
        ap=torchmetrics.AveragePrecision,
        auc=torchmetrics.AUROC,
        f1=torchmetrics.F1Score,
    )

    collection = torchmetrics.MetricCollection(
        {name: metric(task="multiclass", num_classes=n_targets) for name, metric in metrics.items()}
    )

    return torch.nn.ModuleDict(
        {
            "train_collection": collection.clone(),
            "validate_collection": collection.clone(),
            "test_collection": collection.clone(),
        }
    )


@jaxtyped(typechecker=beartype)
def smoothen(targets: Int[torch.Tensor, "N L"], size: int, sigma: float) -> Float[torch.Tensor, "NL _"]:
    """Apply gaussian smoothing to continuous targets with fixed offset for special tokens

    Arguments:
        targets (Int[torch.Tensor, "N L"]): Target label indices.
        size (int): The number of quantiles to smoothen over.
        sigma (float): Gaussian smoothing factor

    Returns:
        Float[torch.Tensor, "N L _"]: Smoothened quantile targets.
    """
    device = targets.device

    N, L = targets.size()

    range_tensor = torch.arange(0, size + len(Tokens), device=device).float()

    # expand and reshape to match the batch and sequence dimensions
    range_tensor = range_tensor.unsqueeze(0).unsqueeze(0).expand(N, L, size + len(Tokens))
    labels_expanded = targets.float().unsqueeze(-1)

    # create gaussian distribution for each label in the sequence
    gaussian = torch.exp(-0.5 * ((range_tensor - labels_expanded) ** 2) / sigma**2)
    gaussian /= gaussian.sum(dim=-1, keepdim=True)

    # one-hot encoding for labels at or below the threshold
    one_hot = torch.zeros_like(gaussian).scatter_(-1, targets.unsqueeze(-1), 1.0)

    # determine which labels are above the threshold
    is_above_threshold = targets >= len(Tokens)

    # prevent gaussian bleeding for labels above the threshold
    start_bleed = torch.zeros_like(targets, dtype=torch.float32) + len(Tokens)
    start_positions = torch.where(is_above_threshold, start_bleed, targets.float())
    prevent_bleed_mask = range_tensor >= start_positions.unsqueeze(-1)

    # re-normalize
    gaussian_masked = gaussian * prevent_bleed_mask.float()
    gaussian_masked /= gaussian_masked.sum(dim=-1, keepdim=True)

    # combine using the condition
    return torch.where(is_above_threshold.unsqueeze(-1), gaussian_masked, one_hot).reshape(N * L, -1)


def pretrain(
    module: "SequenceModule",
    batch: PretrainingData,
    output: OutputData,
    strata: str,
) -> Tensor:
    losses = []

    for field in module.manager.schema.fields:
        values: Tensor = output.reconstructions[field.name]
        targets: TargetField = batch.targets[field.name]

        N, L, T = values.shape

        # smoothen the targets for continuous fields
        if field.type in ["continuous"]:
            labels = smoothen(
                targets=targets.lookup, size=module.params.n_quantiles, sigma=module.params.quantile_smoothing
            )

        elif field.type in ["temporal"]:
            labels = smoothen(
                targets=targets.lookup, size=(366 * 24) + len(Tokens), sigma=module.params.quantile_smoothing
            )

        else:
            assert field.type in ["discrete", "entity"]
            labels = torch.nn.functional.one_hot(targets.lookup.reshape(N * L), num_classes=T).float()

        mask = targets.is_masked.reshape(N * L)
        values = values.reshape(N * L, T)

        loss = cross_entropy(values, labels, reduction="none").masked_select(mask).mean()
        losses.append(loss)
        module.info(f"pretrain-{strata}/{field.name}-loss", loss)

    total_loss = torch.stack(losses).sum()
    module.info(f"pretrain-{strata}/loss", total_loss, prog_bar=(strata == "validate"))
    return total_loss


def finetune(
    module: "SequenceModule",
    batch: SupervisedData,
    output: OutputData,
    strata: str,
) -> Tensor:
    loss = cross_entropy(output.predictions, batch.targets, weight=module.weights)
    module.info(name=f"finetune-{strata}/loss", value=loss, prog_bar=(strata == "validate"))

    probabilities = torch.nn.functional.softmax(output.predictions, dim=1)

    collection = module.metrics[f"{strata}_collection"]

    for name, metric in collection.items():
        metric.update(probabilities, batch.targets)
        module.info(name=f"finetune-{strata}/{name}", value=metric)

    # if (module.global_step % 100 == 0) and (isinstance(module.logger, TensorBoardLogger)):
    #     for index, values in enumerate(probabilities.unbind(dim=1)):
    #         label = module.manager.task.balancer.labels[index]
    #         tag = f'finetune-distribution/{module.manager.schema.target_id}="{label}"'
    #         module.logger.experiment.add_histogram(tag=tag, values=values, global_step=module.global_step)

    return loss


def step(
    self: "SequenceModule",
    batch: SequenceData,
    strata: str,
) -> Tensor:
    """Execute training / inference step

    Args:
        self (SequenceModule): Sequence module
        batch (SequenceData): Training or inference batch
        strata (str): Data strata (one of "train", "validate", "test", "predict")

    Returns:
        Tensor: Loss during training or predictions during inference
    """

    assert strata in ["train", "validate", "test", "predict"]

    output: OutputData = self.forward(batch.inputs)

    if isinstance(batch, PretrainingData):
        return pretrain(module=self, batch=batch, output=output, strata=strata)

    assert isinstance(batch, SupervisedData)

    if strata == "predict":
        return torch.nn.functional.softmax(output.predictions, dim=1)

    else:
        return finetune(module=self, batch=batch, output=output, strata=strata)


def dataloader(self: "SequenceModule", strata: str) -> DataLoader:
    assert strata in ["train", "validate", "test", "predict"]

    pipe = self.pipes[strata]
    self.dataloaders[strata] = loader = DataLoader(
        pipe,
        batch_size=self.params.batch_size,
        num_workers=28,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
    )

    return loader


class SequenceModule(lit.LightningModule):
    def __init__(self, datapath: str, params: Hyperparameters, manager: SystemManager, mode: str):
        super().__init__()

        assert mode in ["pretrain", "finetune", "inference"]

        assert os.path.exists(datapath)
        self.datapath: str = datapath

        assert isinstance(params, Hyperparameters)
        self.params: Hyperparameters = params
        self.save_hyperparameters(params.to_dict())
        self.lr: float = params.learning_rate
        self.mode: str = mode
        self.manager: SystemManager = manager

        self.modular_field_embedder = ModularFieldEmbeddingSystem(params=params, manager=manager)
        self.field_encoder = FieldEncoder(params=params, manager=manager)
        self.event_encoder = EventEncoder(params=params, manager=manager)

        self.event_decoder = EventDecoder(params=params, manager=manager)
        self.decision_head = DecisionHead(params=params, manager=manager)

        self.metrics = create_metrics(manager=manager)
        self.register_buffer("weights", torch.tensor(manager.task.balancer.weights))

        self.dataloaders: dict[str, DataLoader] = {}
        self.pipes: dict[str, datapipes.iter.IterDataPipe] = {}

        self.info = partial(self.log, on_step=False, on_epoch=True, batch_size=self.params.batch_size)

        self.example_input_array = sample = mock(params=params, manager=manager)  # type: ignore
        self.flops_per_batch = measure_flops(self, lambda: self(sample))

    @jaxtyped(typechecker=beartype)
    def forward(self, inputs: TensorDict) -> OutputData:  # type: ignore
        field_mask, event_mask = create_attention_masks(inputs, manager=self.manager)

        fields = self.modular_field_embedder(inputs)
        events = self.field_encoder(fields, mask=field_mask)
        sequence, representations = self.event_encoder(events, mask=event_mask)

        predictions = self.decision_head(sequence)
        reconstructions = self.event_decoder(representations)

        return OutputData.new(sequence=sequence, reconstructions=reconstructions, predictions=predictions)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.mode == "finetune":
            lr = self.lr * self.params.learning_rate_dampener
        else:
            lr = self.lr

        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

    training_step = partialmethod(step, strata="train")  # type: ignore
    validation_step = partialmethod(step, strata="validate")  # type: ignore
    test_step = partialmethod(step, strata="test")  # type: ignore
    predict_step = partialmethod(step, strata="predict")  # type: ignore

    def setup(self, stage: str):
        pipe = partial(stream, datapath=self.datapath, mode=self.mode, params=self.params, manager=self.manager)

        assert stage in ["fit", "validate", "test", "predict"]

        mapping = dict(
            fit=["train", "validate"],
            validate=["validate"],
            test=["test"],
            predict=["predict"],
        )

        for strata in mapping[stage]:
            split = strata if strata != "predict" else "test"
            self.pipes[strata] = pipe(split=split)

    def teardown(self, stage: str):
        assert stage in ["fit", "validate", "test", "predict"]

        mapping = dict(
            fit=["train", "validate"],
            validate=["validate"],
            test=["test"],
            predict=["predict"],
        )

        for strata in mapping[stage]:
            del self.pipes[strata]
            del self.dataloaders[strata]

    train_dataloader = partialmethod(dataloader, strata="train")  # type: ignore
    val_dataloader = partialmethod(dataloader, strata="validate")  # type: ignore
    test_dataloader = partialmethod(dataloader, strata="test")  # type: ignore
    predict_dataloader = partialmethod(dataloader, strata="predict")  # type: ignore

    # def show(self):
    #     memory = humanize.naturalsize(complexity(self.params))
    #     flops = humanize.metric(self.flops_per_batch, "Flops")