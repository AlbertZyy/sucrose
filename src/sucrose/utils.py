from typing import Any, Union, Optional, Mapping, Dict, Iterable

from .sucrose_logger import logger

__all__ = ["lookup", "lookup_path"]


def mapping_copy(data: Mapping[str, Any], /) -> Dict[str, Any]:
    """Copy a mapping recursively."""
    copied = {}

    for key, val in data.items():
        if isinstance(val, Mapping):
            copied[key] = mapping_copy(val)
        else:
            copied[key] = val

    return copied


def lookup(
        data: Mapping[str, Any],
        field: Optional[str] = None,
        *,
        sep: str = "/",
        copy: bool = False,
        default: Any = None
    ) -> Union[Mapping[str, Any], Any]:
    """Lookup the given field path in a mapping.

    Args:
        data (Mapping[str, Any]): The mapping to lookup.
        field (str | None, optional): The field path to lookup.
            Return the input mapping if `None`.
        sep (str, optional): The separator to use. Defaults to "/".
        copy (bool, optional): Whether to copy recursively if the result is a mapping.
            Defaults to False.

    """
    if field is None:
        return mapping_copy(data) if copy else data

    path = field.split(sep)

    return lookup_path(data, path, copy=copy, default=default)


def lookup_path(
        data: Mapping[str, Any],
        path: Iterable[str] = (),
        *,
        copy: bool = False,
        default: Any = None
    ) -> Union[Mapping[str, Any], Any]:
    for key in path:
        if isinstance(data, Mapping):
            if key in data:
                data = data[key]
            else:
                logger.info(f"{data} does not have {key}.")
                return default
        else:
            logger.warning(f"{data} is not a mapping, cannot get {key}.")
            return default

    if copy and isinstance(data, Mapping):
        return mapping_copy(data)
    else:
        return data