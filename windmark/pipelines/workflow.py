from functools import partial

import flytekit as fk

from windmark.core.schema import Field, Hyperparameters

from windmark.tasks import (
    sanitize,
    digest,
    parse,
    preprocess,
    rebalance,
    fit,
    manager
)

@fk.workflow
def pipeline(
    ledger_path: str,
    fields: list[Field],
    params: Hyperparameters,
):

    ledger = sanitize(ledger=ledger_path)

    parsed_fields = fk.map_task(partial(parse, ledger=ledger))(field=fields)

    centroids = fk.map_task(partial(digest, ledger=ledger, slice_size=10_000))(field=parsed_fields)

    balancer = rebalance(ledger=ledger, params=params)
    
    manager(ledger=ledger, shard_size=1, balancer=balancer, params=params)

    lifestreams = preprocess(ledger=ledger, fields=parsed_fields, balancer=balancer)

    fit(dataset=lifestreams, fields=parsed_fields, params=params, centroids=centroids, balancer=balancer)

if __name__ == "__main__":
    pipeline()
