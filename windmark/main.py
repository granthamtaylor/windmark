from windmark.core.structs import Hyperparameters
from windmark.pipelines.workflow import pipeline
from windmark.core.managers import SchemaManager, SplitManager

if __name__ == "__main__":
    ledger = "/home/grantham/windmark/data/quarter_ledger.parquet"

    split = SplitManager(
        train=0.5,
        validate=0.25,
        test=0.25,
    )

    schema = SchemaManager(
        sequence_id="sequence_id",
        event_id="event_id",
        order_by="event_order",
        target_id="target",
        use_chip="discrete",
        merchant_state="discrete",
        merchant_city="discrete",
        merchant_name="entity",
        mcc="discrete",
        amount="continuous",
        timedelta="continuous",
        timestamp="temporal",
    )

    params = Hyperparameters(max_epochs=2, n_steps=500, batch_size=8)

    pipeline(schema=schema, ledger_path=ledger, params=params, split=split)
