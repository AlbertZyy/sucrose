
from typing import Optional

from .const import *
from .header import ProjectHeader, get_current_project
from .logger import start_pytorch_tensorboard


def start_project(work_dir: str, name: str):
    return ProjectHeader(work_dir, name)

def step():
    get_current_project().step()

### Checkpoints

def save_ckpts(epoch: int, data):
    """"""
    get_current_project().save_ckpts(epoch, data)

def load_ckpts(epoch: Optional[int] = None, *, auto_read_step=True):
    """"""
    get_current_project().load_ckpts(epoch, auto_read_step)
