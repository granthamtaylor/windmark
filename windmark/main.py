import windmark as wm

ledger = "/home/grantham/windmark/data/mini-ledger.parquet"

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
    timestamp="temporal",
    card="entity",
    mcc="discrete",
    has_bad_pin="discrete",
    has_bad_zipcode="discrete",
    has_bad_card_number="discrete",
    has_insufficient_balance="discrete",
    has_bad_expiration="discrete",
    has_technical_glitch="discrete",
    has_bad_cvv="discrete",
    timedelta="continuous",
    tenure="continuous",
)

params = wm.Hyperparameters(
    n_pretrain_steps=300,
    n_finetune_steps=50,
    n_context=192,
    batch_size=128,
    d_field=64,
    max_pretrain_epochs=512,
    max_finetune_epochs=256,
    learning_rate=0.0001,
    patience=16,
)

wm.train(datapath=ledger, schema=schema, params=params, split=split)
