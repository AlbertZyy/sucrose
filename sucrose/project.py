
import os
import re
import json
from typing import Dict, Any, Optional

from .const import *
from .ckpt import load_state_dict_impl, save_state_dict_impl
from .logs import start_pytorch_tensorboard_impl
from .sucrose_logger import logger


class Project():
    """Provide projects to manage file paths and names."""
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
        self.load_config()

    def __del__(self):
        if self._local_epoch != 0:
            logger.warning(f"There are still {self._local_epoch} epochs that "
                           "are not saved as checkpoint files by `save_state_dict()`. ")

    @property
    def CKPTS_DIR(self):
        return os.path.join(self.WORK_DIR, CKPTS_FOLDER, self.PROJECT)

    @property
    def LOGS_DIR(self):
        return os.path.join(self.WORK_DIR, LOGS_FOLDER, self.PROJECT)

    def _make_ckpt_name(self, epoch: int):
        return f"{self.PROJECT}_{self.EPOCH_PREFIX}{epoch}{self.CKPTS_EXT}"

    ### Config

    def load_config(self):
        log_file = os.path.join(self.LOGS_DIR, "config.json")

        if not os.path.exists(log_file):
            return None

        with open(log_file, 'r') as f:
            config_data = json.load(f)

        self._config_data.update(config_data)
        logger.info(f"config file loaded from {log_file}")

        return config_data

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

    def epoch_range(self, num: int, /):
        start = self.find_latest_epoch()
        return range(start, start + num)

    ### Ckpts

    def find_latest_epoch(self) -> int:
        if not os.path.exists(self.CKPTS_DIR):
            return 0

        all_ckpts = os.listdir(self.CKPTS_DIR)
        max_epoch = 0

        for name in all_ckpts:
            res = re.match(f'{self.PROJECT}_{self.EPOCH_PREFIX}([0-9]*){self.CKPTS_EXT}', name)
            if res is None:
                continue
            else:
                epoch = int(res.group(1))
                if epoch > max_epoch:
                    max_epoch = epoch

        return max_epoch

    def load_state_dict(self, epoch: Optional[int] = None, *,
        loader_kwds: Dict[str, Any] = {},
        **state_dict: Any
    ):
        if epoch is None:
            epoch = self.find_latest_epoch()
        file_name = self._make_ckpt_name(epoch)
        extra_data, step = load_state_dict_impl(
            self.CKPTS_DIR, file_name, loader_kwds=loader_kwds, **state_dict
        )
        self.num_steps = step
        return extra_data

    def save_state_dict(self, interval: int, *,
        save_step=True,
        extra_kwds: Dict[str, Any] = {},
        **state_dict: Any
    ):
        self._local_epoch += 1
        if self._local_epoch < interval:
            return None

        self._local_epoch = 0
        epoch = self.find_latest_epoch() + interval
        file_name = self._make_ckpt_name(epoch)
        step = self.num_steps if save_step else None
        return save_state_dict_impl(
            self.CKPTS_DIR, file_name, step, extra_kwds=extra_kwds, **state_dict
        )

    ### Logs

    def start_pytorch_tensorboard(self, **kwargs):
        return start_pytorch_tensorboard_impl(self.LOGS_DIR, **kwargs)


def get_current_project() -> Project:
    result = get_current('project')

    if result is None:
        raise RuntimeError("Start a project before getting current project.")

    return result

def auto_get_project(proj: Optional[Project] = None, /) -> Project:
    if proj is None:
        return get_current_project()
    else:
        return proj
