from functools import partial

import flytekit as fl

from source.tasks import (
    read,
    digest,
    parse,
    preprocess,
    parameterize,
    rebalance,
    train,
    export,
)

@fl.workflow
def pipeline():

    fieldreqs = [
        dict(name="use_chip", dtype="discrete"),
        dict(name="merchant_state", dtype="discrete"),
        dict(name="merchant_city", dtype="discrete"),
        dict(name="mcc", dtype="discrete"),
        dict(name="amount", dtype="continuous"),
        dict(name="timedelta", dtype="continuous"),
    ]

    df = read(datapath="/home/grantham/Desktop/windmark/data/ledger.parquet")

    fields = fl.map_task(partial(parse, ledger=df))(fieldreq=fieldreqs)

    digests = fl.map_task(partial(digest, ledger=df, n_slices=10_000))(field=fields)

    lifestreams = preprocess(ledger=df, fields=fields, n_shards=10)

    balancer = rebalance(ledger=df, kappa=0.1)

    params = parameterize(fields=fields, params={})

    module = train(dataset=lifestreams, params=params, digests=digests, balancer=balancer)
    
    
    # export(module=module)

if __name__ == "__main__":
    pipeline()
