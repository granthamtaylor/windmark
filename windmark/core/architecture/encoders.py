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
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from torchdata import datapipes

from windmark.core.managers import SystemManager
from windmark.core.operators import collate, stream, mock
from windmark.core.constructs.tensorfields import TargetField
from windmark.core.constructs.general import Hyperparameters
from windmark.core.constructs.packages import PretrainingData, SupervisedData, OutputData, SequenceData
from windmark.core.architecture.embedders import FieldInterface
from windmark.core.architecture.custom import LinearWarmupCosineAnnealingLR


class LearnedTensor(torch.nn.Module):
    """
    A class representing a learned tensor.

    Args:
        *sizes (int): The sizes of each dimension of the tensor.

    Attributes:
        tensor (torch.Tensor): The learned tensor.

    """

    def __init__(self, *sizes: int):
        """
        Initializes a LearnedTensor object.

        Args:
            *sizes (int): The sizes of each dimension of the tensor.

        Raises:
            AssertionError: If any dimension is not an integer or is less than 1.

        """
        super().__init__()

        for dim in sizes:
            assert isinstance(dim, int), "each dim must be of type int"
            assert dim >= 1, "each dim must be greater than 0"

        tensor = torch.normal(mean=0.0, std=1e-4, size=tuple(sizes))

        self.tensor = torch.nn.Parameter(tensor)

    @jaxtyped(typechecker=beartype)
    def forward(self) -> Float[torch.Tensor, "..."]:
        """
        Performs a forward pass through the learned tensor.

        Returns:
            torch.Tensor: The learned tensor.

        """
        return self.tensor


class ModularFieldEmbeddingSystem(torch.nn.Module):
    """
    A module for performing field embedding in a modular manner.

    Args:
        params (Hyperparameters): The hyperparameters for the embedding system.
        manager (SystemManager): The system manager for accessing the schema and field information.

    Attributes:
        embedders (torch.nn.ModuleDict): A dictionary of field embedders.

    """

    def __init__(self, params: Hyperparameters, manager: SystemManager):
        super().__init__()

        embedders: dict[str, torch.nn.Module] = {}

        for field in manager.schema.fields:
            embedder = FieldInterface.embedder(field)
            embedders[field.name] = embedder(params=params, manager=manager, field=field)

        self.embedders = torch.nn.ModuleDict(embedders)

    @jaxtyped(typechecker=beartype)
    def forward(self, inputs: TensorDict) -> tuple[Float[torch.Tensor, "_N L Fd C"], Float[torch.Tensor, "_N Fs FdC"]]:
        """
        Forward pass of the embedding system.

        Args:
            inputs (TensorDict): The input tensor dictionary.

        Returns:
            tuple[Float[torch.Tensor, "_N L Fd C"], Float[torch.Tensor, "_N Fs FdC"]]: The dynamic and static field embeddings.

        """

        dynamic = []
        static = []

        for field in self.embedders.keys():
            embedder = self.embedders[field]
            embedding = embedder(inputs[field])
            if embedder.type.is_static:
                static.append(embedding)
            else:
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
    Encoder module for dynamic fields in the windmark architecture.

    Args:
        params (Hyperparameters): The hyperparameters for the encoder.
        manager (SystemManager): The system manager for the windmark architecture.

    Attributes:
        encoder (torch.nn.TransformerEncoder): The transformer encoder layer.
        positional (LearnedTensor): The learned positional tensor.

    """

    def __init__(self, params: Hyperparameters, manager: SystemManager):
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
    def forward(self, dynamic: Float[torch.Tensor, "_N L Fd C"]) -> Float[torch.Tensor, "_N L FdC"]:
        """
        Forward pass of the dynamic field encoder.

        Args:
            dynamic (Float[torch.Tensor, "_N L Fd C"]): The input dynamic field tensor.

        Returns:
            Float[torch.Tensor, "_N L FdC"]: The encoded dynamic field tensor.

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
    EventEncoder is a module that encodes events and static features using a TransformerEncoder.

    Args:
        params (Hyperparameters): The hyperparameters for the encoder.
        manager (SystemManager): The system manager for accessing the schema.

    Attributes:
        encoder (torch.nn.TransformerEncoder): The TransformerEncoder layer.
        positional (LearnedTensor): The learned positional tensor.
        class_token (LearnedTensor): The learned class token tensor.

    """

    def __init__(self, params: Hyperparameters, manager: SystemManager):
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
        events: Float[torch.Tensor, "_N L FdC"],
        static: Float[torch.Tensor, "_N Fs FdC"],
    ) -> tuple[
        Float[torch.Tensor, "_N FdC"],
        Float[torch.Tensor, "_N L FdC"],
        Float[torch.Tensor, "_N Fs FdC"],
    ]:
        """
        Forward pass of the encoder module.

        Args:
            events (Float[torch.Tensor, "_N L FdC"]): Input events tensor of shape (N, L, FdC).
            static (Float[torch.Tensor, "_N Fs FdC"]): Input static tensor of shape (N, Fs, FdC).

        Returns:
            tuple[Float[torch.Tensor, "_N FdC"], Float[torch.Tensor, "_N L FdC"], Float[torch.Tensor, "_N Fs FdC"]]:
                - sequence: Encoded sequence tensor of shape (N, FdC).
                - encoded_events: Encoded events tensor of shape (N, L, FdC).
                - encoded_static: Encoded static tensor of shape (N, Fs, FdC).
        """
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
    EventDecoder module for decoding events based on input fields and sequences.
    """

    def __init__(self, params: Hyperparameters, manager: SystemManager):
        """
        Initialize the EventDecoder module.

        Args:
            params (Hyperparameters): The hyperparameters for the module.
            manager (SystemManager): The system manager for accessing schema and fields.
        """
        super().__init__()

        projections = {}

        for field in manager.schema.dynamic:
            tensorfield = FieldInterface.tensorfield(field)

            projections[field.name] = torch.nn.Linear(
                len(manager.schema.dynamic) * params.d_field * 2,
                tensorfield.get_target_size(params=params, manager=manager, field=field),
            )

        self.projections = torch.nn.ModuleDict(projections)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        fields: Float[torch.Tensor, "_N L FdC"],
        sequence: Float[torch.Tensor, "_N FdC"],
    ) -> Annotated[TensorDict, Float[torch.Tensor, "_N L _T"]]:
        """
        Forward pass of the EventDecoder module.

        Args:
            fields (Float[torch.Tensor, "_N L FdC"]): Input fields tensor.
            sequence (Float[torch.Tensor, "_N FdC"]): Input sequence tensor.

        Returns:
            Annotated[TensorDict, Float[torch.Tensor, "_N L _T"]]: Decoded events tensor.
        """
        N, L, FdC = fields.shape

        batched = fields.reshape(N * L, FdC)
        expanded = sequence.repeat(L, 1)
        inputs = torch.cat((batched, expanded), dim=1)

        events = {}

        for field, projection in self.projections.items():
            events[field] = projection(inputs).reshape(N, L, -1)

        # N, L, ?
        return TensorDict(events, batch_size=N)


class StaticFieldDecoder(torch.nn.Module):
    """
    Decoder module for static fields in the windmark architecture.

    Args:
        params (Hyperparameters): The hyperparameters for the model.
        manager (SystemManager): The system manager for the windmark architecture.

    Attributes:
        projections (torch.nn.ModuleDict): A dictionary of linear projections for each static field.

    """

    def __init__(self, params: Hyperparameters, manager: SystemManager):
        super().__init__()

        projections = {}

        for field in manager.schema.static:
            tensorfield = FieldInterface.tensorfield(field)

            projections[field.name] = torch.nn.Linear(
                len(manager.schema.dynamic) * params.d_field * 2,
                tensorfield.get_target_size(params=params, manager=manager, field=field),
            )

        self.projections = torch.nn.ModuleDict(projections)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        fields: Float[torch.Tensor, "_N Fs FdC"],
        sequence: Float[torch.Tensor, "_N FdC"],
    ) -> Annotated[TensorDict, Float[torch.Tensor, "_N _T"]]:
        """
        Performs the forward pass of the StaticFieldDecoder.

        Args:
            fields (Float[torch.Tensor, "_N Fs FdC"]): The input tensor representing the static fields.
            sequence (Float[torch.Tensor, "_N FdC"]): The input tensor representing the sequence.

        Returns:
            TensorDict[Float[torch.Tensor, "_N _T"]]: A dictionary of output tensors for each field.

        """

        N, Fs, FdC = fields.shape

        events = {}

        split = torch.split(fields, 1, dim=1)

        for index, (field, projection) in enumerate(self.projections.items()):
            inputs = torch.cat((split[index].squeeze(dim=1), sequence), dim=1)

            events[field] = projection(inputs)

        # N, ?
        return TensorDict(events, batch_size=N)


class DecisionHead(torch.nn.Module):
    """
    A class representing the decision head of a neural network model.

    Args:
        params (Hyperparameters): The hyperparameters for the model.
        manager (SystemManager): The system manager for the model.

    Attributes:
        mlp (torch.nn.Sequential): The multi-layer perceptron for the decision head.
    """

    def __init__(self, params: Hyperparameters, manager: SystemManager):
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

    @jaxtyped(typechecker=beartype)
    def forward(self, inputs: Float[torch.Tensor, "_N FdC"]) -> Float[torch.Tensor, "_N T"]:
        """
        Defines the forward pass of the DecisionHead.

        Args:
            inputs (Float[torch.Tensor, "_N FdC"]): The input tensor.

        Returns:
            Float[torch.Tensor, "_N T"]: The output of the MLP.
        """

        return self.mlp(inputs)


@beartype
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
def pretrain(
    module: "SequenceModule",
    batch: PretrainingData,
    output: OutputData,
    strata: str,
) -> torch.Tensor:
    """
    Pretrains the sequence module using the given batch of pretraining data and output data.

    Args:
        module (SequenceModule): The sequence module to be pretrained.
        batch (PretrainingData): The batch of pretraining data.
        output (OutputData): The output data generated by the module.
        strata (str): The strata identifier for logging purposes.

    Returns:
        torch.Tensor: The total loss incurred during pretraining.
    """

    dynamic_losses = []
    static_losses = []

    for field in module.manager.schema.dynamic:
        values: torch.Tensor = output.decoded_events[field.name]
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
        values: torch.Tensor = output.decoded_static_fields[field.name]
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


@jaxtyped(typechecker=beartype)
def finetune(
    module: "SequenceModule",
    batch: SupervisedData,
    output: OutputData,
    strata: str,
) -> torch.Tensor:
    """
    Fine-tunes the given module using the provided batch data and output predictions.

    Args:
        module (SequenceModule): The module to be fine-tuned.
        batch (SupervisedData): The batch data containing inputs and targets.
        output (OutputData): The output predictions from the module.
        strata (str): The strata identifier for tracking metrics.

    Returns:
        torch.Tensor: The loss value after fine-tuning.

    """
    loss = cross_entropy(output.predictions, batch.targets, weight=module.weights)
    module.info(name=f"finetune-{strata}/loss", value=loss, prog_bar=(strata == "validate"))

    probabilities = torch.nn.functional.softmax(output.predictions, dim=1)

    collection = module.metrics[f"{strata}_collection"]

    for name, metric in collection.items():
        metric.update(probabilities, batch.targets)
        module.info(name=f"finetune-{strata}/{name}", value=metric)

    return loss


@jaxtyped(typechecker=beartype)
def step(
    self: "SequenceModule",
    batch: SequenceData,
    batch_idx: int,
    strata: str,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Execute training / inference step

    Args:
        self (SequenceModule): Sequence module
        batch (SequenceData): Training or inference batch
        batch_idx (int): Batch index
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


@beartype
def dataloader(self: "SequenceModule", strata: str) -> DataLoader:
    """
    Create and return a DataLoader object for the specified strata.

    Args:
        self (SequenceModule): The SequenceModule instance.
        strata (str): The strata for which to create the DataLoader. Must be one of ["train", "validate", "test", "predict"].

    Returns:
        DataLoader: The DataLoader object for the specified strata.
    """
    assert strata in ["train", "validate", "test", "predict"]

    pipe = self.pipes[strata]
    self.dataloaders[strata] = loader = DataLoader(
        pipe,
        batch_size=self.params.batch_size,
        num_workers=self.params.n_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=False,
    )

    return loader


class SequenceModule(lit.LightningModule):
    """
    SequenceModule is a PyTorch Lightning module that represents a sequence model for encoding and decoding sequences of events and fields.

    Args:
        datapath (str): The path to the data.
        params (Hyperparameters): The hyperparameters for the model.
        manager (SystemManager): The system manager for the model.
        mode (str): The mode of operation for the model. Can be one of "pretrain", "finetune", or "inference".

    Attributes:
        datapath (str): The path to the data.
        params (Hyperparameters): The hyperparameters for the model.
        lr (float): The learning rate for the model.
        mode (str): The mode of operation for the model.
        manager (SystemManager): The system manager for the model.
        modular_field_embedder (ModularFieldEmbeddingSystem): The modular field embedder for the model.
        dynamic_field_encoder (DynamicFieldEncoder): The dynamic field encoder for the model.
        event_encoder (EventEncoder): The event encoder for the model.
        static_field_decoder (StaticFieldDecoder): The static field decoder for the model.
        event_decoder (EventDecoder): The event decoder for the model.
        decision_head (DecisionHead): The decision head for the model.
        metrics (Metrics): The metrics for the model.
        weights (torch.Tensor): The weights for the model.
        dataloaders (dict[str, DataLoader]): The dataloaders for the model.
        pipes (dict[str, datapipes.iter.IterDataPipe]): The data pipes for the model.
        info (Callable): A partial function for logging information.
        example_input_array (torch.Tensor): An example input array for the model.
        flops_per_batch (float): The number of floating point operations per batch for the model.

    Methods:
        forward(inputs: TensorDict) -> OutputData: Performs a forward pass of the model.
        configure_optimizers() -> torch.optim.Optimizer: Configures the optimizers for the model.
        setup(stage: str): Sets up the data pipes for the specified stage.
        teardown(stage: str): Tears down the data pipes for the specified stage.
        train_dataloader() -> DataLoader: Returns the dataloader for the training stage.
        val_dataloader() -> DataLoader: Returns the dataloader for the validation stage.
        test_dataloader() -> DataLoader: Returns the dataloader for the testing stage.
        predict_dataloader() -> DataLoader: Returns the dataloader for the prediction stage.
    """

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

        self.example_input_array = mock(params=params, manager=manager)
        self.flops_per_batch = measure_flops(self, lambda: self.forward(self.example_input_array))

    @jaxtyped(typechecker=beartype)
    def forward(self, inputs: TensorDict) -> OutputData:  # type: ignore
        """
        Forward pass of the encoder module.

        Args:
            inputs (TensorDict): Input tensor dictionary containing the input data.

        Returns:
            OutputData: Output data containing the encoded sequence, decoded events,
                        decoded static fields, and predictions.
        """
        dynamic_fields, static_fields = self.modular_field_embedder(inputs)
        events = self.dynamic_field_encoder(dynamic_fields)
        sequence, encoded_events, encoded_static_fields = self.event_encoder(events, static_fields)

        predictions = self.decision_head(sequence)
        decoded_events = self.event_decoder(encoded_events, sequence)
        decoded_static_fields = self.static_field_decoder(encoded_static_fields, sequence)

        return OutputData.new(
            sequence=sequence,
            decoded_events=decoded_events,
            decoded_static_fields=decoded_static_fields,
            predictions=predictions,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer and scheduler for training the model.

        Returns:
            Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
            A tuple containing the optimizer and scheduler objects.
        """
        if self.mode == "finetune":
            lr = self.lr * self.params.learning_rate_dampener
            max_epochs = self.params.max_finetune_epochs
        else:
            lr = self.lr
            max_epochs = self.params.max_pretrain_epochs

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=self.params.n_epochs_frozen,
            max_epochs=max_epochs,
        )

        return [optimizer], [scheduler]

    training_step = partialmethod(step, strata="train")  # type: ignore
    validation_step = partialmethod(step, strata="validate")  # type: ignore
    test_step = partialmethod(step, strata="test")  # type: ignore
    predict_step = partialmethod(step, strata="predict")  # type: ignore

    def setup(self, stage: str):
        """
        Set up the encoder for a specific stage.

        Args:
            stage (str): The stage for which to set up the encoder. Must be one of ["fit", "validate", "test", "predict"].

        Raises:
            AssertionError: If the stage is not one of the allowed values.

        """
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
        """
        Teardown method for cleaning up resources after a specific stage.

        Args:
            stage (str): The stage for which resources need to be cleaned up. Must be one of ["fit", "validate", "test", "predict"].

        Raises:
            AssertionError: If an invalid stage is provided.

        """
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
