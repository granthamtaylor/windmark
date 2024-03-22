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

        self.outpath = "/home/grantham/windmark/data/predictions.parquet"
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

        if self._destination_file_exists:
            fastparquet.write(self.outpath, df, append=True)

        else:
            print(df)
            fastparquet.write(self.outpath, df)
            self._destination_file_exists = True
