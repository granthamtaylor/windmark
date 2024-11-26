# Copyright Grantham Taylor.

import copy
from typing import Any, Callable, Concatenate, ParamSpec, TypeVar
from functools import partial, wraps

import flytekit
from rich.panel import Panel
from rich.console import Console
from rich.pretty import Pretty


P = ParamSpec("P")
T = TypeVar("T")


# basically, I want the docstring for `flyte.task` to be available for users to see
# this is "copying" the docstring from `flyte.task` to functions wrapped by `forge`
# more details here: https://github.com/python/typing/issues/270


def forge(source: Callable[Concatenate[Any, P], T]) -> Callable[[Callable], Callable[Concatenate[Any, P], T]]:
    def wrapper(target: Callable) -> Callable[Concatenate[Any, P], T]:
        @wraps(source)
        def wrapped(self, *args: P.args, **kwargs: P.kwargs) -> T:
            return target(self, *args, **kwargs)

        return wrapped

    return wrapper


def inherit(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    old = copy.deepcopy(old)
    new = copy.deepcopy(new)

    for key, value in new.items():
        if key in old:
            if isinstance(value, dict):
                old[key] = inherit(old[key], value)
            else:
                old[key] = value
        else:
            old[key] = value

    return old


class TaskEnvironment:
    @forge(flytekit.task)
    def __init__(self, **overrides: Any) -> Any:
        _overrides: dict[str, Any] = {}
        for key, value in overrides.items():
            if key == "_task_function":
                raise KeyError("Cannot override task function")

            _overrides[key] = value

        self.overrides = _overrides

    @forge(flytekit.task)
    def update(self, **overrides: Any) -> None:
        self.overrides = inherit(self.overrides, overrides)

    @forge(flytekit.task)
    def extend(self, **overrides: Any) -> "TaskEnvironment":
        return self.__class__(**inherit(self.overrides, overrides))

    @forge(flytekit.task)
    def __call__(self, _task_function: Callable | None = None, /, **overrides) -> Callable:
        # no additional overrides are passed
        if _task_function is not None:
            if callable(_task_function):
                return partial(flytekit.task, **self.overrides)(_task_function)

            else:
                raise ValueError("The first positional argument must be a callable")

        # additional overrides are passed
        else:

            def inner(_task_function: Callable) -> Callable:
                inherited = inherit(self.overrides, overrides)

                return partial(flytekit.task, **inherited)(_task_function)

            return inner

    def show(self) -> None:
        console = Console()

        console.print(Panel.fit(Pretty(self.overrides)))

    task = __call__
