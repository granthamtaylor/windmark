from functools import partial

import flytekit as fl

from source.tasks import (
    fieldreq,
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

    schema = {
        "use_chip": "discrete",
        "merchant_state": "discrete",
        "merchant_city": "discrete",
        "mcc": "discrete",
        "amount": "continuous",
        "timedelta": "continuous",
    }
    
    fieldreqs = fieldreq(schema=schema)

    df = read(datapath="/home/grantham/Desktop/windmark/data/ledger.subsample.parquet")

    fields = fl.map_task(partial(parse, ledger=df))(fieldreq=fieldreqs)

    digests = fl.map_task(partial(digest, ledger=df, n_slices=10_000))(field=fields)

    lifestreams = preprocess(ledger=df, fields=fields, n_shards=10)

    params = parameterize(fields=fields, params={})

    balancer = rebalance(ledger=df, params=params)

    train(dataset=lifestreams, params=params, digests=digests, balancer=balancer)

if __name__ == "__main__":
    pipeline()
