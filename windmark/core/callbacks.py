import os
from typing import Sequence, Optional

from beartype import beartype
from jaxtyping import jaxtyped
import pyarrow.parquet as pq
import lightning.pytorch as lit
from lightning.pytorch import callbacks
import polars as pl
import torch

from windmark.core.constructs.packages import SupervisedData


class ParquetBatchWriter(callbacks.BasePredictionWriter):
    """
    A callback for writing predictions and representations to a Parquet file in batches.

    Args:
        path (str | os.PathLike): The path to the Parquet file.

    Attributes:
        path (str): The path to the Parquet file.
        schema: The schema of the Parquet file.
        writer: The Parquet writer object.
    """

    def __init__(self, path: str | os.PathLike):
        super().__init__("batch")

        self.path: str = str(path)
        self.schema = None
        self.writer = None

    @jaxtyped(typechecker=beartype)
    def write_on_batch_end(
        self,
        trainer: lit.Trainer,
        pl_module: lit.LightningModule,
        output: tuple[torch.Tensor, torch.Tensor],
        batch_indices: Optional[Sequence[int]],
        batch: SupervisedData,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """
        Called when the predict epoch ends to write the batch predictions and representations to the Parquet file.

        Args:
            trainer (lit.Trainer): The Lightning Trainer object.
            pl_module (lit.LightningModule): The Lightning Module object.
            output (tuple[torch.Tensor, torch.Tensor]): The output tuple containing the predictions and representations.
            batch_indices (Optional[Sequence[int]]): The indices of the batch.
            batch (SupervisedData): The batch data.
            batch_idx (int): The index of the batch.
            dataloader_idx (int): The index of the dataloader.

        Returns:
            None
        """

        predictions, representations = output

        table = (
            pl.DataFrame(
                {
                    "meta": batch.meta,
                    "targets": batch.targets.cpu().detach().numpy(),
                    "predictions": predictions.float().cpu().detach().numpy(),
                    "representations": representations.float().cpu().detach().numpy(),
                }
            )
            .select(
                pl.col("representations"),
                pl.col("predictions"),
                pl.col("targets"),
                pl.col("meta").list.first().alias(pl_module.manager.schema.sequence_id),
                pl.col("meta").list.last().alias(pl_module.manager.schema.event_id),
            )
            .to_arrow()
        )

        if self.writer is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(self.path, self.schema)

        self.writer.write_table(table)

    def on_predict_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        """
        Called at the end of the prediction loop to close the Parquet writer.

        Args:
            trainer (lit.Trainer): The Lightning Trainer object.
            pl_module (lit.LightningModule): The Lightning Module object.

        Returns:
            None
        """
        if self.writer:
            self.writer.close()
            self.writer = None


class ThawedFinetuning(callbacks.BaseFinetuning):
    """
    Callback class for performing thawed finetuning during training.

    Args:
        transition (int): The epoch at which to transition from frozen to unfrozen layers.
    """

    def __init__(self, transition: int):
        super().__init__()

        self.transition = transition

    def freeze_before_training(self, pl_module: lit.LightningModule):
        """
        Freezes the specified layers before training.

        Args:
            pl_module (lit.LightningModule): The LightningModule instance.
        """
        self.freeze(
            [
                pl_module.modular_field_embedder,
                pl_module.dynamic_field_encoder,
                pl_module.event_encoder,
                pl_module.event_decoder,
                pl_module.static_field_decoder,
            ]
        )

    def finetune_function(
        self,
        pl_module: lit.LightningModule,
        epoch: int,
        optimizer: torch.optim.Optimizer,
    ):
        """
        Performs finetuning during training.

        Args:
            pl_module (lit.LightningModule): The LightningModule instance.
            epoch (int): The current epoch.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
        """
        if epoch == self.transition:
            modules = [
                pl_module.modular_field_embedder,
                pl_module.dynamic_field_encoder,
                pl_module.event_encoder,
            ]

            self.unfreeze_and_add_param_group(
                modules=modules,
                optimizer=optimizer,
                train_bn=True,
            )
