
import os
import re
from typing import Dict, Any, Optional, Callable
from functools import partial

from .const import *


class ProjectHeader():
    _step: int
    _save_func: Callable[[Any, str], Any]
    _load_func: Callable[[str], Dict[str, Any]]

    def __init__(self,
        work_dir: str, project: str, *,
        epoch_prefix: str = 'e',
        backend: str = 'pytorch',
        ckpts_ext: Optional[str] = None,
        save_extra_kwds: Dict[str, Any] = {},
        load_extra_kwds: Dict[str, Any] = {},
    ):
        self.PROJECT = project
        self.WORK_DIR = work_dir
        self.EPOCH_PREFIX = epoch_prefix
        self._step = 0

        if backend == 'pytorch':
            import torch
            self._save_func = partial(torch.save, **save_extra_kwds)
            self._load_func = partial(torch.load, **load_extra_kwds)
            self.CKPTS_EXT = '.pt' if ckpts_ext is None else ckpts_ext
        else:
            raise ValueError

        set_current('project', self)

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
        if isinstance(step, int):
            self._step = step
        raise TypeError(f"Step should be an int, but got {step.__class__.__name__}.")

    ### Checkpoints

    def make_ckpt_name(self, epoch: int):
        return f"{self.PROJECT}_{self.EPOCH_PREFIX}{epoch}{self.CKPTS_EXT}"

    def find_latest_epoch(self) -> int:
        all_ckpts = os.listdir(self.CKPTS_DIR)
        max_epoch = 0

        for name in all_ckpts:
            res = re.match(f'{self.PROJECT}_{self.EPOCH_PREFIX}([0-9]*){self.CKPTS_EXT}', name)
            if res is None:
                continue
            else:
                epoch = int(res.group(0))
                if epoch > max_epoch:
                    max_epoch = epoch

        return max_epoch

    def save_ckpts(self, epoch: int, data: Dict[str, Any]):
        os.makedirs(self.CKPTS_DIR, exist_ok=True)
        file_name = os.path.join(self.CKPTS_DIR, self.make_ckpt_name(epoch))
        self._save_func(data, file_name)

    def load_ckpts(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.find_latest_epoch()

        file_name = os.path.join(self.CKPTS_DIR, self.make_ckpt_name(epoch))
        data_loaded = self._load_func(file_name)

        return data_loaded


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
