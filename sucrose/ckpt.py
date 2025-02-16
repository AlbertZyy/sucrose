
import os
from typing import Dict, Mapping, Any, Protocol, Optional, runtime_checkable

from .const import *
from .header import auto_get_project, ProjectHeader
from .sucrose_logger import logger


def save_pt_file(epoch: int, data: Dict[str, Any], *,
                 project: Optional[ProjectHeader] = None):
    from torch import save
    proj = auto_get_project(project)
    os.makedirs(proj.CKPTS_DIR, exist_ok=True)
    file_name = os.path.join(proj.CKPTS_DIR, proj.make_ckpt_name(epoch))
    save(data, file_name)
    logger.info(f"Checkpoint saved to {file_name}")


def load_pt_file(epoch: Optional[int] = None, *,
                 project: Optional[ProjectHeader] = None,
                 loader_kwds: Dict[str, Any] = {}):
    from torch import load
    proj = auto_get_project(project)

    if not os.path.exists(proj.CKPTS_DIR):
        raise FileExistsError(f"Can not find the ckpt directory.")

    if epoch is None:
        epoch = proj.find_latest_epoch()

    file_name = os.path.join(proj.CKPTS_DIR, proj.make_ckpt_name(epoch))

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


def save_state_dict(
    epoch: int, *,
    save_step: bool = True,
    project: Optional[ProjectHeader] = None,
    extra_kwds: Dict[str, Any] = {},
    **state_dict: SupportsStateDict
) -> None:
    """Save state dict to a file."""
    data_to_save = {}
    proj = auto_get_project(project)

    if save_step:
        data_to_save[STEP_KEY] = proj._step

    if extra_kwds:
        data_to_save[EXTRA_KEY] = extra_kwds

    for key, value in state_dict.items():
        if key in data_to_save:
            logger.warning(f"Key '{key}' already exists.")
        data_to_save[key] = value.state_dict()

    if len(data_to_save) > 0:
        save_pt_file(epoch, data_to_save, project=proj)


def load_state_dict(
    epoch: Optional[int] = None, *,
    project: Optional[ProjectHeader] = None,
    loader_kwds: Dict[str, Any] = {},
    load_step: bool = True,
    **state_dict: SupportsStateDict
) -> Dict[str, Any]:
    proj = auto_get_project(project)
    data_loaded = load_pt_file(epoch, project=proj, loader_kwds=loader_kwds)

    if data_loaded is None:
        return None

    if not isinstance(data_loaded, dict):
        raise TypeError("State dicts are expected to be dict, "
                        f"but got {data_loaded.__class__.__name__}")

    if load_step and STEP_KEY in data_loaded:
        proj.set_current_step(data_loaded[STEP_KEY])

    for key, obj in state_dict.items():
        if key in data_loaded:
            obj.load_state_dict(data_loaded.pop(key))

    if EXTRA_KEY in data_loaded:
        return data_loaded[EXTRA_KEY]
    return {}
