# Copyright Grantham Taylor.

import importlib
from types import SimpleNamespace

import flytekit as fl

from windmark.orchestration.context import TaskEnvironment

env = TaskEnvironment(
    container_image=fl.ImageSpec(
        requirements="requirements.txt",
        apt_packages=["build-essential"],
    ),
    secret_requests=[fl.Secret(group="windmark", key="WANDB_API_KEY")],
    cache=True,
    cache_version=str(importlib.metadata.version("windmark")),
    retries=3,
)

context = SimpleNamespace(
    default=env,
    lab=env.extend(
        requests=fl.Resources(gpu="1", cpu="32", mem="64Gi"),
        cache_ignore_input_vars=tuple(["label"]),
    ),
)
