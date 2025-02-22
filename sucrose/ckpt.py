
import os
from typing import Tuple, Dict, Mapping, Any, Protocol, Optional

from .const import *
from .sucrose_logger import logger


def _save_pt_file(ckpts_dir: str, file_name: str, data: Dict[str, Any]):
    from torch import save
    os.makedirs(ckpts_dir, exist_ok=True)
    file_name = os.path.join(ckpts_dir, file_name)
    save(data, file_name)
    logger.info(f"Checkpoint saved to {file_name}")


def _load_pt_file(ckpts_dir: str, file_name: str, **loader_kwds):
    from torch import load
    file_name = os.path.join(ckpts_dir, file_name)

    if os.path.exists(file_name):
        data_loaded = load(file_name, **loader_kwds)
        logger.info(f"Checkpoint loaded from {file_name}")
        return data_loaded
    else:
        logger.info(f"No checkpoint exists, loading skipped.")
        return None


class SupportsStateDict(Protocol):
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state_dict: Mapping[str, Any]): ...


def save_state_dict_impl(
    ckpts_dir: str,
    ckpt_file: str,
    step: Optional[int],
    *,
    extra_kwds: Dict[str, Any] = {},
    **state_dict: SupportsStateDict
) -> None:
    data_to_save = {}

    if step is not None:
        data_to_save[STEP_KEY] = step

    if extra_kwds:
        data_to_save[EXTRA_KEY] = extra_kwds

    for key, value in state_dict.items():
        if key in data_to_save:
            logger.warning(f"Key '{key}' already exists.")
        data_to_save[key] = value.state_dict()

    if len(data_to_save) > 0:
        os.makedirs(ckpts_dir, exist_ok=True)
        _save_pt_file(ckpts_dir, ckpt_file, data_to_save)


def load_state_dict_impl(
    ckpts_dir: str,
    ckpt_file: str,
    *,
    loader_kwds: Dict[str, Any] = {},
    **state_dict: SupportsStateDict
) -> Tuple[Dict[str, Any], int]:
    data_loaded = _load_pt_file(ckpts_dir, ckpt_file, **loader_kwds)

    if data_loaded is None:
        return None

    if not isinstance(data_loaded, dict):
        raise TypeError("State dicts are expected to be dict, "
                        f"but got {data_loaded.__class__.__name__}")

    if STEP_KEY in data_loaded:
        step = data_loaded[STEP_KEY]
    else:
        step = 0

    for key, obj in state_dict.items():
        if key in data_loaded:
            obj.load_state_dict(data_loaded.pop(key))

    if EXTRA_KEY in data_loaded:
        return data_loaded[EXTRA_KEY], step

    return {}, step
