import windmark as wm

if __name__ == "__main__":
    ledger = "/home/grantham/windmark/data/ledger.parquet"

    split = wm.SequenceSplitter(
        train=0.60,
        validate=0.20,
        test=0.20,
    )

    schema = wm.Schema.create(
        sequence_id="customer_id",
        event_id="transaction_id",
        order_by="order_id",
        target_id="is_fraud",
        merchant_state="discrete",
        merchant_city="discrete",
        use_chip="discrete",
        amount="continuous",
        merchant_name="entity",
        card="entity",
        timestamp="temporal",
        mcc="discrete",
        has_bad_pin="discrete",
        has_bad_zipcode="discrete",
        has_bad_card_number="discrete",
        has_insufficient_balance="discrete",
        has_bad_expiration="discrete",
        has_technical_glitch="discrete",
        has_bad_cvv="discrete",
    )

    params = wm.Hyperparameters(
        n_pretrain_steps=2000,
        n_finetune_steps=200,
        batch_size=192,
        d_field=48,
        max_epochs=256,
        n_epochs_frozen=16,
        learning_rate=0.00005,
    )

    wm.train(datapath=ledger, schema=schema, params=params, split=split)
