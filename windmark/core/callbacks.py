import os

import pyarrow.parquet as pq

import lightning.pytorch as lit
import msgspec
import polars as pl
import torch

from windmark.core.structs import InferenceData


class ParquetBatchWriter(lit.callbacks.BasePredictionWriter):
    def __init__(self, outpath: str | os.PathLike):
        super().__init__("batch")

        self.outpath = outpath
        self.schema = None
        self.writer = None

    def write_on_batch_end(
        self,
        trainer: lit.Trainer,
        module: lit.LightningModule,
        prediction: torch.Tensor,
        batch_indices: list[int] | None,
        batch: InferenceData,
        batch_index: int,
        dataloader_index: int,
    ) -> None:
        """
        Called when the predict epoch ends.
        """

        array = prediction.float().cpu().detach().numpy()

        table = (
            pl.DataFrame(
                {
                    "meta": batch.meta,
                    "predictions": array,
                }
            )
            .select(
                pl.col("predictions").map_elements(lambda x: msgspec.json.encode(x.to_list()), return_dtype=pl.String),
                sequence_id=pl.col("meta").list.first(),
                event_id=pl.col("meta").list.last(),
            )
            .to_arrow()
        )

        if self.writer is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(self.outpath, self.schema)

        self.writer.write_table(table)

    def on_predict_end(self, trainer: lit.Trainer, module: lit.LightningModule):
        """
        Called at the end of the prediction loop to close the Parquet writer.
        """
        if self.writer:
            self.writer.close()
            self.writer = None


class ThawedFinetuning(lit.callbacks.BaseFinetuning):
    def __init__(self, transition: int):
        super().__init__()

        self._unfreeze_at_epoch = transition

    def freeze_before_training(self, module: lit.LightningModule):
        self.freeze(
            [
                module.modular_field_embedder,
                module.field_encoder,
                module.event_encoder,
                module.event_decoder,
            ]
        )

    def finetune_function(
        self,
        module: lit.LightningModule,
        epoch: int,
        optimizer: torch.optim.Optimizer,
    ):
        if epoch == self._unfreeze_at_epoch:
            modules = [
                module.modular_field_embedder,
                module.field_encoder,
                module.event_encoder,
            ]

            self.unfreeze_and_add_param_group(
                modules=modules,
                optimizer=optimizer,
                train_bn=True,
            )
