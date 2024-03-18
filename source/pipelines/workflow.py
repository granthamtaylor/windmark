from functools import partial
from pathlib import Path

import flytekit as fk

from source.tasks import (
    sanitize,
    fieldreq,
    digest,
    parse,
    preprocess,
    parameterize,
    rebalance,
    train,
    export,
)

@fk.workflow
def pipeline():

    schema = {
        "use_chip": "discrete",
        "merchant_state": "discrete",
        "merchant_city": "discrete",
        "mcc": "discrete",
        "amount": "continuous",
        "timedelta": "continuous",
    }
    
    fieldreqs = fieldreq(schema=schema)

    ledger = sanitize(ledger="/home/grantham/windmark/data/ledger.subsample.parquet")

    fields = fk.map_task(partial(parse, ledger=ledger))(fieldreq=fieldreqs)

    centroids = fk.map_task(partial(digest, ledger=ledger, n_slices=10_000))(field=fields)

    params = parameterize(fields=fields, params={})

    balancer = rebalance(ledger=ledger, params=params)

    lifestreams = preprocess(ledger=ledger, fields=fields, shard_size=1, balancer=balancer)

    train(dataset=lifestreams, params=params, centroids=centroids, balancer=balancer)

if __name__ == "__main__":
    pipeline()
