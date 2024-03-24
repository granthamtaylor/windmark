from windmark.core.structs import Hyperparameters, Schema
from windmark.pipelines.workflow import pipeline

if __name__ == "__main__":
    ledger = "/home/grantham/windmark/data/ledger.parquet"

    schema = Schema(
        use_chip="discrete",
        merchant_state="discrete",
        merchant_city="discrete",
        merchant_name="entity",
        mcc="discrete",
        amount="continuous",
        timedelta="continuous",
        timestamp="temporal",
    )

    params = Hyperparameters(
        n_fields=len(schema), max_epochs=256, pretrain_sample_rate=0.005, finetune_sample_rate=0.02
    )

    pipeline(fields=schema.fields, ledger_path=ledger, params=params)
