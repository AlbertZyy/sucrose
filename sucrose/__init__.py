
from typing import Optional

from .const import *
from .header import get_current_project, auto_get_project
from .header import ProjectHeader as Proj
from .logs import start_pytorch_tensorboard
from .ckpt import load_state_dict, save_state_dict
from .config import read_config
from .sucrose_logger import logger


def start_project(
        work_dir: str, name: str, *,
        epoch_prefix: str = 'e',
        ckpts_ext: Optional[str] = None
    ):
    return Proj(work_dir, name, epoch_prefix=epoch_prefix, ckpts_ext=ckpts_ext)


### Training

def step(project: Optional[Proj] = None, /):
    auto_get_project(project).step()

def get_current_step(project: Optional[Proj] = None, /):
    auto_get_project(project).get_current_step()

def epoch_range(length: int, *, project: Optional[Proj] = None):
    start = auto_get_project(project).find_latest_epoch() + 1
    return range(start, start + length)
