from source.tasks import Field

fields: list[Field] = [
    Field("transaction_type", "discrete", n_levels=4),
    Field("amount", "continuous"),
    Field("merchant_category", "discrete", n_levels=18),
    Field("is_foreign", "discrete", n_levels=2),
    Field("is_online", "discrete", n_levels=2),
]
