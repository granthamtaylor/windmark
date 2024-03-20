from functools import partial

import flytekit as fk

from source.core.schema import Field, Hyperparameters

from source.tasks import (
    sanitize,
    digest,
    parse,
    preprocess,
    rebalance,
    fit,
    manager
)

@fk.workflow
def pipeline():

    fields = [
        Field(use_chip="discrete"),
        Field(merchant_state="discrete"),
        Field(merchant_city="discrete"),
        Field(merchant_name="entity"),
        Field(mcc="discrete"),
        Field(amount="continuous"),
        Field(timedelta="continuous"),
    ]
    
    params = Hyperparameters(n_fields=len(fields))

    ledger = sanitize(ledger="/home/grantham/windmark/data/ledger.parquet")

    fields = fk.map_task(partial(parse, ledger=ledger))(field=fields)

    centroids = fk.map_task(partial(digest, ledger=ledger, slice_size=10_000))(field=fields)

    balancer = rebalance(ledger=ledger, params=params)
    
    manager(ledger=ledger, shard_size=1, balancer=balancer, params=params)

    lifestreams = preprocess(ledger=ledger, fields=fields, balancer=balancer)

    fit(dataset=lifestreams, fields=fields, params=params, centroids=centroids, balancer=balancer)

if __name__ == "__main__":
    pipeline()
