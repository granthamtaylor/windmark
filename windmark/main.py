from windmark.pipelines.workflow import pipeline
from windmark.core.schema import Schema, Hyperparameters


if __name__ == '__main__':
    
    ledger = ("/home/grantham/windmark/data/ledger.subsample.parquet")

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
        n_fields=len(schema)
    )
    
    pipeline(fields=schema.fields, ledger_path=ledger, params=params)
