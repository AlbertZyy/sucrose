
from typing import Optional, Dict, Any

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
        backend: str = 'pytorch',
        ckpts_ext: Optional[str] = None,
        save_extra_kwds: Dict[str, Any] = {},
        load_extra_kwds: Dict[str, Any] = {}):
    return Proj(work_dir, name)


### Training

def step(project: Optional[Proj] = None, /):
    auto_get_project(project).step()

def get_current_step(project: Optional[Proj] = None, /):
    auto_get_project(project).get_current_step()

def epoch_range(length: int, *, project: Optional[Proj] = None):
    start = auto_get_project(project).find_latest_epoch() + 1
    return range(start, start + length)

### Checkpoints

def save_ckpts(epoch: int, data, *, project: Optional[Proj] = None):
    """Save a checkpoint file for the current project."""
    auto_get_project(project).save_ckpts(epoch, data)

def load_ckpts(epoch: Optional[int] = None, *, auto_read_step=True,
               project: Optional[Proj] = None):
    """Load a checkpoint file of the current project."""
    return auto_get_project(project).load_ckpts(epoch, auto_read_step=auto_read_step)
