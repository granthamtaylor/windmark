from functools import partial

import flytekit as fk

from source.core.schema import Schema, Hyperparameters

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

    schema = Schema(
        use_chip="discrete",
        merchant_state="discrete",
        merchant_city="discrete",
        merchant_name="entity",
        mcc="discrete",
        amount="continuous",
        timedelta="continuous",
        timestamp="temporal",
    )
    
    params = Hyperparameters(n_fields=len(schema))

    ledger = sanitize(ledger="/home/grantham/windmark/data/ledger.subsample.parquet")

    fields = fk.map_task(partial(parse, ledger=ledger))(field=schema.fields)

    centroids = fk.map_task(partial(digest, ledger=ledger, slice_size=10_000))(field=fields)

    balancer = rebalance(ledger=ledger, params=params)
    
    manager(ledger=ledger, shard_size=1, balancer=balancer, params=params)

    lifestreams = preprocess(ledger=ledger, fields=fields, balancer=balancer)

    fit(dataset=lifestreams, fields=fields, params=params, centroids=centroids, balancer=balancer)

if __name__ == "__main__":
    pipeline()
