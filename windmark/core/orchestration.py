from functools import partial
import importlib

import flytekit

version = str(importlib.metadata.version("windmark"))

task = partial(flytekit.task, cache=True, cache_version=version)
