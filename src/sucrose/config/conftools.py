
__all__ = ["partial_config"]

from typing import Any, TypeVar, Generic
from functools import partial
from collections.abc import Mapping, Callable
import inspect

_R = TypeVar("_R")


def _get_keyword_params(target: Callable, /) -> tuple[list[str], list[bool], bool]:
    if not hasattr(target, "__dict__"):
        raise TypeError("func must have __dict__ attribute")

    if isinstance(target, type):
        if "__init__" in target.__dict__:
            target = target.__init__
        elif "__new__" in target.__dict__:
            target = target.__new__
        else:
            raise TypeError("target must have __init__ or __new__ method "
                            "to deduce keyword parameters")

    sig = inspect.signature(target)
    results = []
    has_default = []
    has_var_keyword = False

    for name, param in sig.parameters.items():
        if param.kind in (param.POSITIONAL_ONLY, param.VAR_POSITIONAL):
            continue
        if param.kind == param.VAR_KEYWORD:
            has_var_keyword = True
            continue

        results.append(name)
        has_default.append(param.default != param.empty)

    return results, has_default, has_var_keyword


class partial_config(partial, Generic[_R]):
    def __new__(cls, func: Callable[..., _R], /, dataset: Mapping[str, Any]):
        keywords = {}
        names_all, has_default_all, has_var_keyword = _get_keyword_params(func)

        if has_var_keyword: # all keyword arguments are acceptable
            keywords.update(dataset)
            return super().__new__(cls, func, **keywords)

        for name, has_default in zip(names_all, has_default_all):
            try:
                keywords[name] = dataset[name]
            except KeyError:
                if not has_default:
                    raise ValueError(f"key {name!r} is missing in dataset")

        return super().__new__(cls, func, **keywords)

    __call__: Callable[..., _R]
