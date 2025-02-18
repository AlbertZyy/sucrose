
import os
import json
import inspect
from typing import (
    Optional, Generic, TypeVar,
    Callable, List
)

from .header import auto_get_project
from .header import ProjectHeader as Proj
from .sucrose_logger import logger

_MT = TypeVar('_MT')


def load_config(project: Optional[Proj] = None, /):
    """Load data from config.json and update the config of the project."""
    proj = auto_get_project(project)
    log_file = os.path.join(proj.LOGS_DIR, "config.json")

    if not os.path.exists(log_file):
        return None

    with open(log_file, 'r') as f:
        config_data = json.load(f)

    proj.update_config(**config_data)
    logger.info(f"config file loaded from {log_file}")

    return config_data


def enable_config(field: Optional[str] = None, /):
    """Allow Sucrose to manage the parameters required for initialization.

    Position-only args are not supported as the config data is stored in a dict.

    Args:
        field (str | None, optional): Key in the config dict. Use the whole config dict if `None`.
    """
    def _wrapper(module: type[_MT]):
        return _ConfigWrapper(module, field)
    return _wrapper


def _get_KEYWORD_params(func: Callable) -> List[str]:
    sig = inspect.signature(func)
    accepts_keyword = []

    for name, param in sig.parameters.items():
        if name == 'self':
            continue

        if param.kind in (param.POSITIONAL_ONLY, param.VAR_POSITIONAL):
            continue

        accepts_keyword.append(name)

    return accepts_keyword


class _ConfigWrapper(Generic[_MT]):
    def __init__(self, module_class: type[_MT], field: Optional[str] = None,
                 verbose=True):
        self._module_class = module_class
        self._field = field
        self._verbose = verbose
        self._keyword_bound = _get_KEYWORD_params(module_class.__init__)
        self._proj = None

    def set_project(self, project: Proj, /):
        self._proj = project
        return self

    def __call__(self, *args, **kwds) -> _MT:
        proj = auto_get_project(self._proj)
        data = proj.CONFIG
        key = self._field

        if key is not None:
            if key in data:
                data = data[self._field]
            else:
                logger.warning(f"Key '{key}' is not found in the config dict.")

        if not isinstance(data, dict):
            raise TypeError(f"config data is required to be a dict")

        for param_key in tuple(data.keys()):
            if param_key not in self._keyword_bound:
                data.pop(param_key)
                logger.debug(f"Ignored key '{param_key}' which is not supported "
                             f"by {self._module_class.__name__}.")

        if self._verbose:
            print(f"Sucrose: add the following args for {self._module_class.__name__} "
                  f"in project {proj.PROJECT}:")
            for param_key, param_val in data.items():
                print(f"\t{param_key}:\t{param_val}")

        data.update(kwds)

        return self._module_class(*args, **data)
