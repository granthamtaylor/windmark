import math
import os
from collections import OrderedDict
from functools import partial, partialmethod
from typing import Annotated

import lightning.pytorch as lit
import torch
import torchmetrics
from beartype import beartype
from jaxtyping import Float, jaxtyped
from lightning.fabric.utilities.throughput import measure_flops
from tensordict import TensorDict
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from torchdata import datapipes

from windmark.core.managers import SystemManager
from windmark.core.operators import collate, stream, mock
from windmark.core.constructs.tensorfields import TargetField
from windmark.core.constructs.general import Hyperparameters
from windmark.core.constructs.interface import FieldInterface
from windmark.core.constructs.packages import PretrainingData, SupervisedData, OutputData, SequenceData


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

        for field in manager.schema.fields:
            embedder = FieldInterface.embedder(field)
            embedders[field.name] = embedder(params=params, manager=manager, field=field)

        self.embedders = torch.nn.ModuleDict(embedders)

    @jaxtyped(typechecker=beartype)
    def forward(self, inputs: TensorDict) -> tuple[Float[Tensor, "N L Fd C"], Float[Tensor, "N Fs FdC"]]:
        """
        Performs the forward pass of the ModularFieldEmbeddingSystem.

        Args:
            inputs (TensorDict[Float[Tensor, "N L"] | Int[Tensor, "N L"]]): The input tensor.

        Returns:
            Float[Tensor, "N L F"]: The embedded fields.
        """

        dynamic = []
        static = []

        for field in self.embedders.keys():
            embedder = self.embedders[field]
            embedding = embedder(inputs[field])
            if embedder.type.is_static:
                static.append(embedding)
            elif not embedder.type.is_static:
                dynamic.append(embedding)

        # N L Fd C
        dynamic_fields = torch.stack(dynamic, dim=-1).permute(0, 1, 3, 2)

        N, L, Fd, C = dynamic_fields.shape

        # N Fs FdC
        if len(static) > 0:
            static_fields = torch.stack(static, dim=-1).permute(0, 2, 1)
        else:
            # empty if static fields are not available
            static_fields = torch.empty(size=(N, 0, Fd * C), device=dynamic_fields.device)

        # (N L Fd C), (N Fs FdC)
        return dynamic_fields, static_fields


class DynamicFieldEncoder(torch.nn.Module):
    """
    DynamicFieldEncoder is a PyTorch module for encoding dynamic fields.
    """

    def __init__(self, params: Hyperparameters, manager: SystemManager):
        """
        Initialize field transformer encoder.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
            manager (SystemManager): The pipeline system manager.
        """

        super().__init__()

        layer = torch.nn.TransformerEncoderLayer(
            d_model=params.d_field,
            nhead=params.n_heads_field_encoder,
            batch_first=True,
            dropout=params.dropout,
            activation="gelu",
        )

        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=params.n_layers_field_encoder)

        self.positional = LearnedTensor(len(manager.schema.dynamic), params.d_field)

    @jaxtyped(typechecker=beartype)
    def forward(self, dynamic: Float[Tensor, "N L Fd C"]) -> Float[Tensor, "N L FdC"]:
        """
        Performs the forward pass of the DynamicFieldEncoder.

        Args:
            inputs (Float[Tensor, "N L Fd C"]): The input tensor.

        Returns:
            Float[Tensor, "N L Fd C"]: The encoded dynamic fields.
        """

        N, L, Fd, C = dynamic.shape

        # NL Fd C
        batched = dynamic.view(N * L, Fd, C) + self.positional()

        # NL Fd C
        events = self.encoder(batched).view(N, L, Fd * C)

        # N L FdC
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

        layer = torch.nn.TransformerEncoderLayer(
            d_model=len(manager.schema.dynamic) * params.d_field,
            nhead=params.n_heads_event_encoder,
            batch_first=True,
            dropout=params.dropout,
            activation="gelu",
        )

        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=params.n_layers_event_encoder)

        Fd = len(manager.schema.dynamic)
        Fs = len(manager.schema.static)

        self.positional = LearnedTensor(1 + params.n_context + Fs, Fd * params.d_field)
        self.class_token = LearnedTensor(1, 1, Fd * params.d_field)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        events: Float[Tensor, "N L FdC"],
        static: Float[Tensor, "N Fs FdC"],
    ) -> tuple[
        Float[Tensor, "N FdC"],
        Float[Tensor, "N L FdC"],
        Float[Tensor, "N Fs FdC"],
    ]:
        N, L, FdC = events.shape
        N, Fs, FdC = static.shape

        # N 1 FdC
        class_token = self.class_token().expand(N, 1, FdC)

        # N L+1+Fs FdC
        concatenated = torch.cat((class_token, events, static), dim=1)

        # N L+1+Fs FdC
        encoded = self.encoder(concatenated + self.positional())

        # (N 1 FdC), (N L FdC), (N Fs FdC)
        sequence, encoded_events, encoded_static = encoded.split((1, L, Fs), dim=1)

        return sequence.squeeze(dim=1), encoded_events, encoded_static


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

        for field in manager.schema.dynamic:
            tensorfield = FieldInterface.tensorfield(field)

            projections[field.name] = torch.nn.Conv1d(
                in_channels=len(manager.schema.dynamic) * params.d_field,
                out_channels=tensorfield.get_target_size(params=params, manager=manager, field=field),
                kernel_size=1,
            )

        self.projections = torch.nn.ModuleDict(projections)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        inputs: Float[Tensor, "N L FdC"],
    ) -> Annotated[TensorDict, Float[Tensor, "N L _"]]:
        """
        Performs the forward pass of the EventDecoder.

        Args:
            inputs (Float[Tensor, "N L FdC"]): The input tensor.

        Returns:
            TensorDict[Float[Tensor, "N L _"]]: A dictionary of output tensors for each field.
        """

        N = inputs.shape[0]

        # N, FdC, L
        permuted = inputs.permute(0, 2, 1)

        events = {}

        for field, projection in self.projections.items():
            events[field] = projection(permuted).permute(0, 2, 1)

        # N, L, ?
        return TensorDict(events, batch_size=N)


class StaticFieldDecoder(torch.nn.Module):
    """
    StaticFieldDecoder is a PyTorch module for decoding masked static fields from their contextualized representations.
    """

    def __init__(self, params: Hyperparameters, manager: SystemManager):
        """
        Initialize the static field decoder.

        Args:
            params (Hyperparameters): The hyperparameters for the architecture.
            manager (SystemManager): The pipeline system manager.
        """

        super().__init__()

        projections = {}

        for field in manager.schema.static:
            tensorfield = FieldInterface.tensorfield(field)

            projections[field.name] = torch.nn.Linear(
                len(manager.schema.dynamic) * params.d_field,
                tensorfield.get_target_size(params=params, manager=manager, field=field),
            )

        self.projections = torch.nn.ModuleDict(projections)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        inputs: Float[Tensor, "N Fs FdC"],
    ) -> Annotated[TensorDict, Float[Tensor, "N _"]]:
        """
        Performs the forward pass of the EventDecoder.

        Args:
            inputs (Float[Tensor, "N L FdC"]): The input tensor.

        Returns:
            TensorDict[Float[Tensor, "N L _"]]: A dictionary of output tensors for each field.
        """

        N, Fs, FdC = inputs.shape

        # N, FsFdC
        # permuted = inputs.view(N, Fs * FdC)

        events = {}

        split = torch.split(inputs, 1, dim=1)

        for index, (field, projection) in enumerate(self.projections.items()):
            events[field] = projection(split[index].squeeze(dim=1))

        # N, ?
        return TensorDict(events, batch_size=N)


class DecisionHead(torch.nn.Module):
    """
    A PyTorch module representing a decision head. This module is used to map the encoded fields to the target classes.
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
                    math.ceil(math.log(len(manager.schema.dynamic) * params.d_field, params.head_shape_log_base)),
                ),
            )
        )

        hidden.reverse()

        dims = [len(manager.schema.dynamic) * params.d_field, *hidden, manager.task.n_targets]

        layers = []

        norm = torch.nn.utils.parametrizations.weight_norm

        for index, sizes in enumerate(zip(dims[:-1], dims[1:])):
            if index < len(dims) - 2:
                activation = torch.nn.GELU()
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
        Defines the forward pass of the DecisionHead.

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


def pretrain(
    module: "SequenceModule",
    batch: PretrainingData,
    output: OutputData,
    strata: str,
) -> Tensor:
    dynamic_losses = []
    static_losses = []

    for field in module.manager.schema.dynamic:
        values: Tensor = output.decoded_events[field.name]
        targets: TargetField = batch.targets[field.name]

        N, L, T = values.shape

        tensorfield = FieldInterface.tensorfield(field)

        labels = tensorfield.postprocess(values=values, targets=targets, params=module.params)

        mask = targets.is_masked.reshape(N * L)
        values = values.reshape(N * L, T)

        loss = cross_entropy(values, labels, reduction="none").mul(mask).sum().div(mask.sum().clamp(min=1))
        dynamic_losses.append(loss)
        module.info(f"pretrain-dynamic-{strata}/{field.name}", loss)

    p_mask_dynamic = (
        module.params.p_mask_field
        + module.params.p_mask_event
        - module.params.p_mask_field * module.params.p_mask_event
    )

    scalar = (module.params.d_field * module.params.p_mask_static) / (module.params.n_context * p_mask_dynamic)

    for field in module.manager.schema.static:
        values: Tensor = output.decoded_static_fields[field.name]
        targets: TargetField = batch.targets[field.name]

        N, T = values.shape

        tensorfield = FieldInterface.tensorfield(field)

        labels = tensorfield.postprocess(values=values, targets=targets, params=module.params)

        mask = targets.is_masked.reshape(N)
        values = values.reshape(N, T)

        loss = cross_entropy(values, labels, reduction="none").mul(mask).sum().div(mask.sum().clamp(min=1)).mul(scalar)
        static_losses.append(loss)
        module.info(f"pretrain-static-{strata}/{field.name}", loss)

    dynamic_loss = torch.stack(dynamic_losses).sum()
    module.info(f"pretrain-total-{strata}/dynamic", dynamic_loss)

    if len(module.manager.schema.static) > 0:
        static_loss = torch.stack(static_losses).sum()
        module.info(f"pretrain-total-{strata}/static", static_loss)
    else:
        static_loss = torch.zeros_like(dynamic_loss)

    total_loss = dynamic_loss + static_loss

    module.info(f"pretrain-total-{strata}/loss", total_loss, prog_bar=(strata == "validate"))

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
) -> Tensor | tuple[Tensor, Tensor]:
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
        predictions = torch.nn.functional.softmax(output.predictions, dim=1)
        return predictions, output.sequence

    else:
        return finetune(module=self, batch=batch, output=output, strata=strata)


def dataloader(self: "SequenceModule", strata: str) -> DataLoader:
    assert strata in ["train", "validate", "test", "predict"]

    pipe = self.pipes[strata]
    self.dataloaders[strata] = loader = DataLoader(
        pipe,
        batch_size=self.params.batch_size,
        num_workers=self.params.n_workers,
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
        self.dynamic_field_encoder = DynamicFieldEncoder(params=params, manager=manager)
        self.event_encoder = EventEncoder(params=params, manager=manager)

        self.static_field_decoder = StaticFieldDecoder(params=params, manager=manager)
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
        dynamic_fields, static_fields = self.modular_field_embedder(inputs)
        events = self.dynamic_field_encoder(dynamic_fields)
        sequence, encoded_events, encoded_static_fields = self.event_encoder(events, static_fields)

        predictions = self.decision_head(sequence)
        decoded_events = self.event_decoder(encoded_events)
        decoded_static_fields = self.static_field_decoder(encoded_static_fields)

        return OutputData.new(
            sequence=sequence,
            decoded_events=decoded_events,
            decoded_static_fields=decoded_static_fields,
            predictions=predictions,
        )

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
