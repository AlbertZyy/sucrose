
__all__ = ["config_from_data", "config_from_project"]

from typing import Any, TypeVar
from collections.abc import Mapping, Callable

from ..project import Project, auto_get_project
from ..sucrose_logger import logger
from .configs import *
from .conftools import partial_config

_MT = TypeVar('_MT')


def config_from_data(
    prefix: str,
    domain: str,
    data: Mapping[str, Mapping[str, Any]],
):
    dataset = {item[0].rsplit("/", 1)[-1]: item[1]
               for item in find_all(data, domain, prefix)}
    if dataset:
        data_repr = "\n".join([f"{k!r}\t = {v!r}" for k, v in dataset.items()])
        logger.info(f"Configuring {prefix!r} from domain {domain!r} "
                    f"with following args:\n{data_repr}")
    else:
        logger.info(f"No config found for {prefix!r} in domain {domain!r}.")

    def _wrapper(obj: Callable[..., _MT]):
        return partial_config(obj, dataset)

    return _wrapper


def config_from_project(
    prefix: str,
    proj: Project | None = None,
):
    """Allow Sucrose to manage the parameters required for calling.

    Position-only args are not supported as the config data is stored in a dict.
    """
    proj = auto_get_project(proj)
    config: dict[str, dict[str, Any]] = proj.CONFIG

    return config_from_data(prefix, domain=proj.PROJECT, data=config)
