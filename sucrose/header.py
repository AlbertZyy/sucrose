
import os
from typing import Any, Optional

from .const import *
from .sucrose_logger import logger


class ProjectHeader():
    """Provide headers for projects to manage file paths and names."""
    _step : int
    _local_epoch : int # changed by `save_state_dict`

    def __init__(self,
        work_dir: str, project: str, *,
        epoch_prefix: str = 'e',
        ckpts_ext: Optional[str] = None,
    ):
        self.PROJECT = project
        self.WORK_DIR = work_dir
        self.EPOCH_PREFIX = epoch_prefix
        if ckpts_ext is None:
            ckpts_ext = '.pt'
        self.CKPTS_EXT = ckpts_ext
        self._step = 0 # number of steps finished, index of the next
        self._local_epoch = 0
        self._config_data = {}

        set_current('project', self)
        self._makedir()

    def __del__(self):
        if self._local_epoch != 0:
            logger.warning(f"There are still {self._local_epoch} epochs that "
                           "are not saved as checkpoint files by `save_state_dict()`. ")

    def _makedir(self):
        os.makedirs(self.CKPTS_DIR, exist_ok=True)
        os.makedirs(self.LOGS_DIR, exist_ok=True)

    @property
    def CKPTS_DIR(self):
        return os.path.join(self.WORK_DIR, CKPTS_FOLDER, self.PROJECT)

    @property
    def LOGS_DIR(self):
        return os.path.join(self.WORK_DIR, LOGS_FOLDER, self.PROJECT)

    def _make_ckpt_name(self, epoch: int):
        return f"{self.PROJECT}_{self.EPOCH_PREFIX}{epoch}{self.CKPTS_EXT}"

    ### Config

    def update_config(self, **data: Any):
        self._config_data.update(data)

    @property
    def CONFIG(self):
        return self._config_data.copy()

    ### Training

    def step(self, num: int = 1, /):
        self._step += num

    @property
    def num_steps(self): return self._step
    @num_steps.setter
    def num_steps(self, step: int):
        if not isinstance(step, int):
            raise TypeError(f"Step should be an int, but got {step.__class__.__name__}.")
        self._step = step


def get_current_project() -> ProjectHeader:
    result = get_current('project')

    if result is None:
        raise RuntimeError("Start a project before getting current project.")

    return result

def auto_get_project(proj: Optional[ProjectHeader] = None, /) -> ProjectHeader:
    if proj is None:
        return get_current_project()
    else:
        return proj
