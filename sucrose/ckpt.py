
import os
import re
from typing import Dict, Mapping, Any, Protocol, Optional

from .const import *
from .header import auto_get_project, ProjectHeader
from .sucrose_logger import logger


def find_latest_epoch(proj: ProjectHeader, /) -> int:
    """Find the number of finished epoches, which is alse the index(0-base)
    of the next epoch. Return `0` if no epoch found."""
    if not os.path.exists(proj.CKPTS_DIR):
        return 0

    all_ckpts = os.listdir(proj.CKPTS_DIR)
    max_epoch = 0

    for name in all_ckpts:
        res = re.match(f'{proj.PROJECT}_{proj.EPOCH_PREFIX}([0-9]*){proj.CKPTS_EXT}', name)
        if res is None:
            continue
        else:
            epoch = int(res.group(1))
            if epoch > max_epoch:
                max_epoch = epoch

    return max_epoch


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


def save_state_dict(
    epoch_interval: int = 1,
    project: Optional[ProjectHeader] = None,
    *,
    save_step: bool = True,
    extra_kwds: Dict[str, Any] = {},
    **state_dict: SupportsStateDict
) -> None:
    """Save state dict to a file.

    Args:
        epoch_interval (int): The epoch increase compared to the last saved checkpoint file.
        project (ProjectHeader, optional): Project, defaults to the last created.
        save_step (bool, optional): Let the checkpoint file include step info,
            which grows as the `sucrose.step()` function is called. Defaults to `True`.
        extra_kwds (Dict[str, Any]): Other data.

    Examples:
        ```
        sucrose.save_state_dict(10, model=model, optim=optim)
        ```
        This line may generate a checkpoint file like:
        ```
        {
            "model": {...} # state dict of the model
            "optim": {...} # state dict of the optimizer
            "step": 1000 # for example
        }
        ```
    """
    proj = auto_get_project(project)
    proj._local_epoch += 1

    if proj._local_epoch < epoch_interval:
        return None

    proj._local_epoch = 0
    data_to_save = {}
    epoch = find_latest_epoch(proj) + epoch_interval

    if save_step:
        data_to_save[STEP_KEY] = proj.num_steps

    if extra_kwds:
        data_to_save[EXTRA_KEY] = extra_kwds

    for key, value in state_dict.items():
        if key in data_to_save:
            logger.warning(f"Key '{key}' already exists.")
        data_to_save[key] = value.state_dict()

    if len(data_to_save) > 0:
        _save_pt_file(proj.CKPTS_DIR, proj._make_ckpt_name(epoch), data_to_save)


def load_state_dict(
    epoch: Optional[int] = None,
    project: Optional[ProjectHeader] = None,
    *,
    loader_kwds: Dict[str, Any] = {},
    load_step: bool = True,
    **state_dict: SupportsStateDict
) -> Dict[str, Any]:
    """Load state dict to objects.

    Args:
        epoch (int | None, optional): The epoch number of the checkpoint file to read.
            Use the biggest number found in the folder if `None`. Defaults to `None`.
        project (ProjectHeader, optional): Project, defaults to the last created.
        loader_kwds (Dict[str, Any], optional): Keyword args for the loader function
            like `torch.load`.
        load_step (bool, optional): Read step info (if exists) from file into
            the project header if `True`. Defaults to `True`.

    Examples:
        ```
        sucrose.load_state_dict(model=model, optim=optim)
        ```
        This line is equivelant to:
        ```
        data = torch.load("path/to/checkpoint.pt")
        model.load_state_dict(data['model'])
        optim.load_state_dict(data['optim'])
        ph.num_step = data['step'] # this key is actually `sucrose.const.STEP_KEY`
        ```
    """
    proj = auto_get_project(project)
    if epoch is None:
        epoch = find_latest_epoch(proj)
    data_loaded = _load_pt_file(proj.CKPTS_DIR, proj._make_ckpt_name(epoch), **loader_kwds)

    if data_loaded is None:
        return None

    if not isinstance(data_loaded, dict):
        raise TypeError("State dicts are expected to be dict, "
                        f"but got {data_loaded.__class__.__name__}")

    if load_step and STEP_KEY in data_loaded:
        proj.num_steps = data_loaded[STEP_KEY]

    for key, obj in state_dict.items():
        if key in data_loaded:
            obj.load_state_dict(data_loaded.pop(key))

    if EXTRA_KEY in data_loaded:
        return data_loaded[EXTRA_KEY]
    return {}
