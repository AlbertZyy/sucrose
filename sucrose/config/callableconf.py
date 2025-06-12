
import inspect
from typing import (
    Optional, Generic, TypeVar, Type,
    Callable, List, Dict, Any,
    overload
)

from ..project import Project, auto_get_project
from ..utils import lookup_path
from ..sucrose_logger import logger


_MT = TypeVar('_MT')

@overload
def enable_config(
    fields: str,
    *,
    sep: Optional[str] = "/",
    proj: Optional[Project] = None,
    verbose=True
) -> Callable[[Callable[..., _MT]], "_ConfigWrapper[_MT]"]: ...
@overload
def enable_config(
    *fields: str,
    proj: Optional[Project] = None,
    verbose=True
) -> Callable[[Callable[..., _MT]], "_ConfigWrapper[_MT]"]: ...
def enable_config(
        *fields: str,
        sep: Optional[str] = "/",
        proj: Optional[Project] = None,
        verbose=True
    ):
    """Allow Sucrose to manage the parameters required for calling.

    Position-only args are not supported as the config data is stored in a dict.

    Args:
        *fields (str): Key in the config dict. Use the whole config dict if not given.
        sep (str | None, optional): Separate a single field string to path if not `None`.
            Ignored when multiple field strings are given. Defaults to `"/"`
        proj (Project | None, optional): The project whose config to load kwargs from.
            Always load kwargs from context project in __call__ if `None`.
            Defaults to `None`.
        verbose (bool, optional): Whether to print the kwargs added to the callable.
            Defaults to `True`.
    """
    if len(fields) == 1 and sep is not None:
        path = fields[0].split(sep)
    else:
        path = fields

    def _wrapper(obj: Callable[..., _MT]):
        return _ConfigWrapper(obj, path, proj=proj, verbose=verbose)
    return _wrapper


def _get_KEYWORD_params(func: Callable) -> List[str]:
    is_class = isinstance(func, Type)
    sig = inspect.signature(func.__init__ if is_class else func)
    accepts_keyword = []

    for name, param in sig.parameters.items():
        if is_class and name == 'self':
            continue

        if param.kind in (param.POSITIONAL_ONLY, param.VAR_POSITIONAL):
            continue

        accepts_keyword.append(name)

    return accepts_keyword


class _ConfigWrapper(Generic[_MT]):
    _kwargs : Optional[Dict[str, Any]]

    def __init__(self,
            target: Callable[..., _MT],
            config_path: List[str],
            proj: Optional[Project] = None,
            verbose: bool = True
        ):
        self._target = target
        self._cpath = config_path
        self._verbose = verbose
        self._keyword_bound = _get_KEYWORD_params(target)
        self._proj = proj

        if proj is None:
            self._kwargs = None
        else:
            self._kwargs, _ = self._load_config()

    def _load_config(self):
        proj = auto_get_project(self._proj)
        NoReturnFlag = object()
        kwargs = lookup_path(proj.CONFIG, self._cpath, copy=True, default=NoReturnFlag)

        if kwargs is NoReturnFlag:
            tname = self._target.__name__
            raise RuntimeError(f"config kwargs for {tname} is not found "
                                f"in {proj.PROJECT}:{'/'.join(self._cpath)}")

        if not isinstance(kwargs, dict):
            raise TypeError("config kwargs is required to be a dict, "
                            f"but got {kwargs.__class__.__name__}")

        for param_key in tuple(kwargs.keys()):
            if param_key not in self._keyword_bound:
                kwargs.pop(param_key)
                logger.debug(f"Ignored key '{param_key}' which is not supported "
                             f"by {self._target.__name__}.")
        return kwargs, proj

    def _get_config(self):
        if self._kwargs is None:
            return self._load_config()
        else:
            assert self._proj is not None
            return self._kwargs, self._proj

    def __call__(self, *args, **kwds) -> _MT:
        data, proj = self._get_config()

        if self._verbose:
            print(f"Sucrose: add the following args for {self._target.__name__} "
                  f"in project {proj.PROJECT}:")
            for param_key, param_val in data.items():
                print(f"\t{param_key}:\t{param_val}")

        data.update(kwds)

        return self._target(*args, **data)

    def __getattr__(self, name):
        return getattr(self._target, name)
