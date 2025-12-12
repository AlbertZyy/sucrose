
__all__ = [
    'load_state_dict_impl',
    'save_state_dict_impl',
    'SupportsStateDict'
]

import os
from typing import Any, Protocol, runtime_checkable


def _save_pt_file(ckpts_dir: str, file_name: str, data: dict[str, Any]):
    from torch import save
    os.makedirs(ckpts_dir, exist_ok=True)
    file_name = os.path.join(ckpts_dir, file_name)
    save(data, file_name)


def _load_pt_file(ckpts_dir: str, file_name: str, **loader_kwds):
    from torch import load
    file_name = os.path.join(ckpts_dir, file_name)

    if os.path.exists(file_name):
        data_loaded = load(file_name, **loader_kwds)
        return data_loaded
    else:
        raise FileNotFoundError(f"No checkpoint exists at {file_name}")


@runtime_checkable
class SupportsStateDict(Protocol):
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state_dict: dict[str, Any]) -> Any: ...


def save_state_dict_impl(
    ckpts_dir: str,
    ckpt_file: str,
    **state_dict: SupportsStateDict | Any
) -> None:
    data_to_save = {}

    for key, value in state_dict.items():
        if isinstance(value, SupportsStateDict):
            data_to_save[key] = value.state_dict()
        else:
            data_to_save[key] = value

    if len(data_to_save) > 0:
        os.makedirs(ckpts_dir, exist_ok=True)
        _save_pt_file(ckpts_dir, ckpt_file, data_to_save)


def load_state_dict_impl(
    ckpts_dir: str,
    ckpt_file: str,
    loader_kwds: dict[str, Any] = {},
    **state_dict: SupportsStateDict
) -> dict[str, Any]:
    """
    Load state dicts from a checkpoint file, then load into the given objects
    that supports state dict operations.

    Return the remaining data in the checkpoint as a dict.
    """
    data_loaded = _load_pt_file(ckpts_dir, ckpt_file, **loader_kwds)

    if not isinstance(data_loaded, dict):
        raise TypeError("State dicts are expected to be dict, "
                        f"but got {data_loaded.__class__.__name__}")

    for key, obj in state_dict.items():
        if key in data_loaded:
            obj.load_state_dict(data_loaded.pop(key))

    return data_loaded
