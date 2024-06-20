import windmark as wm

ledger = "/home/grantham/windmark/data/mini-ledger.parquet"

split = wm.SequenceSplitter(train=0.60, validate=0.20, test=0.20)

schema = wm.Schema.create(
    # structural
    sequence_id="customer_id",
    event_id="transaction_id",
    order_by="order_id",
    target_id="is_fraud",
    # fields
    merchant_state="static_discrete",
    merchant_city="static_discrete",
    use_chip="static_discrete",
    amount="continuous",
    merchant_name="entity",
    timestamp="temporal",
    card="entity",
    mcc="discrete",
    # has_bad_pin="discrete",
    # has_bad_zipcode="discrete",
    # has_bad_card_number="discrete",
    # has_insufficient_balance="discrete",
    # has_bad_expiration="discrete",
    # has_technical_glitch="discrete",
    # has_bad_cvv="discrete",
    # timedelta="continuous",
    tenure="static_continuous",
)

params = wm.Hyperparameters(
    n_pretrain_steps=30,
    n_finetune_steps=10,
    n_context=3,
    batch_size=2,
    d_field=64,
    max_pretrain_epochs=8,
    max_finetune_epochs=8,
    n_layers_event_encoder=4,
    n_epochs_frozen=4,
    n_heads_event_encoder=8,
    n_layers_field_encoder=1,
    n_heads_field_encoder=8,
    learning_rate=0.00005,
    patience=4,
)

wm.train(datapath=ledger, schema=schema, params=params, split=split)
