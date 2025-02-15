
from typing import Dict, Mapping, Any, Protocol, Optional

from .const import *
from .header import auto_get_project, ProjectHeader


class SupportsStateDict(Protocol):
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state_dict: Mapping[str, Any]): ...


def save_state_dict(
    epoch: int, *,
    model: Optional[SupportsStateDict] = None,
    optim: Optional[SupportsStateDict] = None,
    extra_kwds: Dict[str, Any] = {},
    save_step: bool = True,
    project: Optional[ProjectHeader] = None
):
    data_to_save = {}
    proj = auto_get_project(project)

    if model is not None:
        data_to_save[MODEL_KEY] = model.state_dict()
    if optim is not None:
        data_to_save[OPTIM_KEY] = optim.state_dict()
    if extra_kwds:
        data_to_save[EXTRA_KEY] = extra_kwds
    if save_step:
        data_to_save[STEP_KEY] = proj._step

    if len(data_to_save) > 0:
        proj.save_ckpts(epoch, data_to_save)


def load_state_dict(
    epoch: Optional[int] = None, *,
    model: Optional[SupportsStateDict] = None,
    optim: Optional[SupportsStateDict] = None,
    load_step: bool = True,
    project: Optional[ProjectHeader] = None
) -> Dict[str, Any]:
    proj = auto_get_project(project)
    data_loaded = proj.load_ckpts(epoch)

    if MODEL_KEY in data_loaded:
        model.load_state_dict(data_loaded[MODEL_KEY])
    if OPTIM_KEY in data_loaded:
        optim.load_state_dict(data_loaded[OPTIM_KEY])
    if load_step and STEP_KEY in data_loaded:
        proj.set_current_step(data_loaded[STEP_KEY])

    if EXTRA_KEY in data_loaded:
        return data_loaded[EXTRA_KEY]
    else:
        return {}
