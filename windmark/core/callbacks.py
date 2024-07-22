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
    def __init__(self, outpath: str | os.PathLike):
        super().__init__("batch")

        self.outpath = outpath
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
        Called when the predict epoch ends.
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
            self.writer = pq.ParquetWriter(self.outpath, self.schema)

        self.writer.write_table(table)

    def on_predict_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule):
        """
        Called at the end of the prediction loop to close the Parquet writer.
        """
        if self.writer:
            self.writer.close()
            self.writer = None


class ThawedFinetuning(callbacks.BaseFinetuning):
    def __init__(self, transition: int):
        super().__init__()

        self.transition = transition

    def freeze_before_training(self, pl_module: lit.LightningModule):
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
