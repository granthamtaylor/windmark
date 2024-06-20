import windmark as wm

ledger = "/home/grantham/windmark/data/mini-ledger.parquet"

split = wm.SequenceSplitter(train=0.60, validate=0.20, test=0.20)

schema = wm.Schema.new(
    # structural
    sequence_id="customer_id",
    event_id="transaction_id",
    order_by="order_id",
    target_id="is_fraud",
    # fields
    merchant_state="categories",
    merchant_city="categories",
    use_chip="categories",
    amount="numbers",
    merchant_name="entities",
    timestamp="timestamps",
    card="entities",
    mcc="categories",
    has_bad_pin="categories",
    has_bad_zipcode="categories",
    has_bad_card_number="categories",
    has_insufficient_balance="categories",
    has_bad_expiration="categories",
    has_technical_glitch="categories",
    has_bad_cvv="categories",
    timedelta="numbers",
    tenure="numbers",
)

params = wm.Hyperparameters(
    n_pretrain_steps=300,
    n_finetune_steps=100,
    n_context=256,
    batch_size=64,
    d_field=64,
    max_pretrain_epochs=128,
    max_finetune_epochs=128,
    n_layers_event_encoder=4,
    n_epochs_frozen=4,
    n_heads_event_encoder=8,
    n_layers_field_encoder=1,
    n_heads_field_encoder=8,
    learning_rate=0.00005,
    patience=4,
)

wm.train(datapath=ledger, schema=schema, params=params, split=split)
