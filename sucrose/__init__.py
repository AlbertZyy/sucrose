
from typing import Optional

from .const import *
from .header import get_current_project, auto_get_project
from .header import ProjectHeader as _Proj
from .logs import start_pytorch_tensorboard
from .ckpt import find_latest_epoch, load_state_dict, save_state_dict
from .config import load_config, enable_config
from .sucrose_logger import logger


def start_project(
        work_dir: str, name: str, *,
        epoch_prefix: str = 'e',
        ckpts_ext: Optional[str] = None,
        load_conf=True
    ):
    """Start a project and return its header.

    Args:
        work_dir (str): Path to the workspace filder.
        name (str): Project name.
        epoch_prefix (str, optional): Prefix for the epoch number in checkpoint
            file names. Defaults to `e`.
        ckpts_ext (str, optional): Extension for ckeckpoint files. Defualts to `.pt`.
        load_conf (bool, optional): Load config.json for this project. This is
            equivalent to `sucrose.load_config()`.

    Returns:
        ProjectHeader.
    """
    proj = _Proj(work_dir, name, epoch_prefix=epoch_prefix, ckpts_ext=ckpts_ext)
    if load_conf:
        load_config(proj)
    return proj


### Training

def step(num: int = 1, /, project: Optional[_Proj] = None):
    auto_get_project(project).step(num)

def get_current_step(project: Optional[_Proj] = None, /):
    auto_get_project(project).get_current_step()

def epoch_iter(num: int, /, *, project: Optional[_Proj] = None):
    start = find_latest_epoch(auto_get_project(project))
    yield from range(start, start + num)
