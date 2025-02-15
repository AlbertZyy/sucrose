
import os
import re
from typing import Dict, Any, Optional, Callable

from .const import *


class ProjectHeader():
    _step: int
    _save_func: Callable[[Any, str], Any]
    _load_func: Callable[[str], Dict[str, Any]]

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
        self._step = 0

        set_current('project', self)
        self._makedir()

    def _makedir(self):
        os.makedirs(self.CKPTS_DIR, exist_ok=True)
        os.makedirs(self.LOGS_DIR, exist_ok=True)

    @property
    def CKPTS_DIR(self):
        return os.path.join(self.WORK_DIR, CKPTS_FOLDER, self.PROJECT)

    @property
    def LOGS_DIR(self):
        return os.path.join(self.WORK_DIR, LOGS_FOLDER, self.PROJECT)

    ### Training

    def step(self): self._step += 1
    def get_current_step(self): return self._step
    def set_current_step(self, step: int):
        if not isinstance(step, int):
            raise TypeError(f"Step should be an int, but got {step.__class__.__name__}.")
        self._step = step

    ### Checkpoints

    def make_ckpt_name(self, epoch: int):
        return f"{self.PROJECT}_{self.EPOCH_PREFIX}{epoch}{self.CKPTS_EXT}"

    def find_latest_epoch(self) -> int:
        """Find the index of the last completed epoch. Return `-1` if no epoch found."""
        if not os.path.exists(self.CKPTS_DIR):
            return -1

        all_ckpts = os.listdir(self.CKPTS_DIR)
        max_epoch = -1

        for name in all_ckpts:
            res = re.match(f'{self.PROJECT}_{self.EPOCH_PREFIX}([0-9]*){self.CKPTS_EXT}', name)
            if res is None:
                continue
            else:
                epoch = int(res.group(1))
                if epoch > max_epoch:
                    max_epoch = epoch

        return max_epoch


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
