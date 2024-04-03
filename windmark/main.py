import windmark as wm

if __name__ == "__main__":
    ledger = "/home/grantham/windmark/data/quarter_ledger.parquet"

    split = wm.SequenceSplitter(
        train=0.70,
        validate=0.15,
        test=0.15,
    )

    schema = wm.Schema.create(
        sequence_id="customer_id",
        event_id="transaction_id",
        order_by="order_id",
        target_id="target",
        use_chip="discrete",
        merchant_state="discrete",
        merchant_city="discrete",
        merchant_name="entity",
        has_bad_pin="discrete",
        has_bad_zipcode="discrete",
        has_bad_card_number="discrete",
        has_insufficient_balance="discrete",
        has_bad_expiration="discrete",
        has_technical_glitch="discrete",
        has_bad_cvv="discrete",
        card="entity",
        mcc="discrete",
        amount="continuous",
        timedelta="continuous",
        timestamp="temporal",
    )

    params = wm.Hyperparameters(
        n_steps=20,
        batch_size=128,
        max_epochs=256,
        d_field=48,
    )

    wm.train(schema=schema, ledger_path=ledger, params=params, split=split)
