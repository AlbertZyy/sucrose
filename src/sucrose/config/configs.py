
__all__ = ["lookup", "find_all"]

from typing import Any
from collections.abc import Mapping, Iterator


def check_data(data: Mapping[str, Mapping[str, Any]]) -> None:
    for key in data:
        if key.startswith("/"):
            raise ValueError(f"domain name cannot start with '/'")


def lookup(
    data: Mapping[str, Mapping[str, Any]],
    domain: str,
    field: str
) -> Any:
    """Lookup the given field path in data."""
    check_data(data)

    if domain in data:
        dom_data = data[domain]

        if field in dom_data:
            return dom_data[field]

    parent_domain = domain.rsplit("/", 1)[0]

    if parent_domain == domain:
        raise KeyError(
            f"can not resolve field {field!r} from the domain {domain!r}"
        )

    return lookup(data, parent_domain, field)


def find_all(
    data: Mapping[str, Mapping[str, Any]],
    domain: str,
    prefix: str,
    exclude: set[str] = set()
) -> Iterator[tuple[str, Any]]:
    check_data(data)
    exclude = exclude.copy()

    if domain in data:
        dom_data = data[domain]
        for field, value in dom_data.items():
            if field.startswith(prefix) and (field not in exclude):
                yield field, value
                exclude.add(field)

    parent_domain = domain.rsplit("/", 1)[0]

    if parent_domain == domain: # if reached the root
        return

    yield from find_all(data, parent_domain, prefix, exclude)
