
__all__ = ["config_from_data"]

from typing import Any, TypeVar
from collections.abc import Mapping, Callable

from ..sucrose_logger import logger
from .configs import *
from .conftools import partial_config

_MT = TypeVar('_MT')


def config_from_data(
    func: Callable[..., _MT],
    /,
    prefix: str,
    domain: str,
    data: Mapping[str, Mapping[str, Any]],
):
    dataset = {item[0].rsplit(".", 1)[-1]: item[1]
               for item in find_all(data, domain, prefix)}
    if dataset:
        data_repr = "\n".join([f"{k!r}\t = {v!r}" for k, v in dataset.items()])
        logger.info(f"Configuring {prefix!r} from domain {domain!r} "
                    f"with following args:\n{data_repr}")
    else:
        logger.info(f"No config found for {prefix!r} in domain {domain!r}.")

    return partial_config(func, dataset)
