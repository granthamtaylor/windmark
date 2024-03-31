import windmark as wm

if __name__ == "__main__":
    ledger = "/home/grantham/windmark/data/quarter_ledger.parquet"

    split = wm.SequenceSplitter(
        train=0.70,
        validate=0.15,
        test=0.15,
    )

    schema = wm.Schema.create(
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

    params = wm.Hyperparameters(
        n_steps=160,
        batch_size=16,
        max_epochs=2,
        n_epochs_frozen=1,
    )

    wm.pipeline(schema=schema, ledger_path=ledger, params=params, split=split)
