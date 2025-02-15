
from typing import Optional, Dict, Any

from .const import *
from .header import ProjectHeader, get_current_project
from .logger import start_pytorch_tensorboard
from .ckpt import load_state_dict, save_state_dict
from .config import read_config


def start_project(
        work_dir: str, name: str, *,
        epoch_prefix: str = 'e',
        backend: str = 'pytorch',
        ckpts_ext: Optional[str] = None,
        save_extra_kwds: Dict[str, Any] = {},
        load_extra_kwds: Dict[str, Any] = {}):
    return ProjectHeader(work_dir, name)


### Training

def step():
    get_current_project().step()

def get_current_step():
    get_current_project().get_current_step()


### Checkpoints

def save_ckpts(epoch: int, data):
    """Save a checkpoint file for the current project."""
    get_current_project().save_ckpts(epoch, data)

def load_ckpts(epoch: Optional[int] = None, *, auto_read_step=True):
    """Load a checkpoint file of the current project."""
    return get_current_project().load_ckpts(epoch, auto_read_step=auto_read_step)
