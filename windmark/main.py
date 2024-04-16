import windmark as wm

if __name__ == "__main__":
    ledger = "/home/grantham/windmark/data/ledger.parquet"

    split = wm.SequenceSplitter(
        train=0.60,
        validate=0.20,
        test=0.20,
    )

    schema = wm.Schema.create(
        # structural
        sequence_id="customer_id",
        event_id="transaction_id",
        order_by="order_id",
        target_id="is_fraud",
        # fields
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
        n_pretrain_steps=800,
        n_finetune_steps=40,
        n_context=160,
        batch_size=256,
        d_field=64,
        max_pretrain_epochs=128,
        max_finetune_epochs=256,
        n_layers_event_encoder=4,
        learning_rate=0.0005,
        patience=4,
    )

    wm.train(datapath=ledger, schema=schema, params=params, split=split)
