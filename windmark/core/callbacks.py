import os

import fastparquet
import lightning.pytorch as lit
import msgspec
import polars as pl
import torch

from windmark.core.structs import InferenceData


class ParquetBatchWriter(lit.callbacks.BasePredictionWriter):
    def __init__(self, outpath: str | os.PathLike):
        super().__init__("batch")

        self.outpath = outpath
        self._destination_file_exists: bool = False

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
        array = prediction.float().cpu().detach().numpy()

        df = (
            pl.DataFrame(
                {
                    "meta": batch.meta,
                    "predictions": array,
                }
            )
            .select(
                pl.col("predictions").map_elements(lambda x: msgspec.json.encode(x.to_list())),
                sequence_id=pl.col("meta").list.first(),
                event_id=pl.col("meta").list.last(),
            )
            .to_pandas()
        )

        fastparquet.write(self.outpath, df, append=self._destination_file_exists)

        if not self._destination_file_exists:
            self._destination_file_exists = True


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
            modules = [module.modular_field_embedder, module.field_encoder, module.event_encoder]

            self.unfreeze_and_add_param_group(
                modules=modules,
                optimizer=optimizer,
                train_bn=True,
            )
