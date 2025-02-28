"""
Sucrose
=======

The sucrose package manages files of pytorch experiment projects.
"""

from typing import Optional, Any, Dict

from .const import *
from .project import get_current_project, auto_get_project
from .project import Project as _Proj
from .model_config import enable_config
from .sucrose_logger import logger


def start_project(
        work_dir: str, name: str, *,
        epoch_prefix: str = 'e',
        ckpts_ext: Optional[str] = None,
    ):
    """Start a project and return its header.

    Args:
        work_dir (str): Path to the workspace filder.
        name (str): Project name.
        epoch_prefix (str, optional): Prefix for the epoch number in checkpoint
            file names. Defaults to `e`.
        ckpts_ext (str, optional): Extension for ckeckpoint files. Defualts to `.pt`.

    Returns:
        Project.
    """
    proj = _Proj(work_dir, name, epoch_prefix=epoch_prefix, ckpts_ext=ckpts_ext)
    return proj


### Training

def step(num: int = 1, /, proj: Optional[_Proj] = None):
    return auto_get_project(proj).step(num)

def get_current_step(proj: Optional[_Proj] = None, /):
    return auto_get_project(proj).num_steps

def epoch_range(num: int, /, *, proj: Optional[_Proj] = None):
    """Create a range object to iterate over epochs."""
    return auto_get_project(proj).epoch_range(num)

### Ckpts

def find_last_epoch(proj: Optional[_Proj] = None, /):
    """Find the number of finished epoches, which is alse the index(0-base)
        of the next epoch. Return `0` if no epoch found."""
    return auto_get_project(proj).find_latest_epoch()

def load_state_dict(
        epoch: Optional[int] = None, *,
        proj: Optional[_Proj] = None,
        loader_kwds: Dict[str, Any] = {},
        **state_dict: Any
    ):
    """Load state dict to objects.

    Args:
        epoch (int | None, optional): The epoch number of the checkpoint file to read.
            Use the biggest number found in the folder if `None`. Defaults to `None`.
        project (ProjectHeader, optional): Project, defaults to the last created.
        loader_kwds (Dict[str, Any], optional): Keyword args for the loader function
            like `torch.load`.
        load_step (bool, optional): Read step info (if exists) from file into
            the project header if `True`. Defaults to `True`.

    Returns:
        Dict: Objects remaining in the dictionary after loading the state dict.
            Or only the data in the dictionary with key `EXTRA_KEY` will be returned
            if `EXTRA_KEY` (a global constant, defaults to `'extra'`) exists.

    Examples:
        ```
        sucrose.load_state_dict(model=model, optim=optim)
        ```
        This line is equivelant to:
        ```
        data = torch.load("path/to/checkpoint.pt")
        model.load_state_dict(data['model'])
        optim.load_state_dict(data['optim'])
        ph.num_step = data['step'] # this key is actually `sucrose.const.STEP_KEY`
        ```
    """
    return auto_get_project(proj).load_state_dict(
        epoch, loader_kwds=loader_kwds, **state_dict
    )

def save_state_dict(
        interval: int, *,
        proj: Optional[_Proj] = None,
        save_step=True,
        extra_kwds: Dict[str, Any] = {},
        **state_dict: Any
    ):
    """Save state dict to `WORK_DIR/CKPTS_FOLDER/PROJECT/FILE_NAME.pt`.

    Args:
        interval (int): The epoch increase compared to the last saved checkpoint file.
            Calling this function will increase the save counter of the project by 1.
            The save operation will be triggered only when the counter reaches the interval,
            and then the counter will be cleared.
        project (ProjectHeader, optional): Project, defaults to the last created.
        save_step (bool, optional): Let the checkpoint file include step info,
            which grows as the `sucrose.step()` function is called. Defaults to `True`.
        extra_kwds (Dict[str, Any]): Other data.

    Examples:
        ```
        sucrose.save_state_dict(10, model=model, optim=optim)
        ```
        This line may generate a checkpoint file like:
        ```
        {
            "model": {...} # state dict of the model
            "optim": {...} # state dict of the optimizer
            "step": 1000 # for example
        }
        ```
    """
    return auto_get_project(proj).save_state_dict(
        interval, save_step=save_step, extra_kwds=extra_kwds, **state_dict
    )

### Logs

def start_pytorch_tensorboard(proj: Optional[_Proj] = None, **kwargs):
    """Start a pytorch tensorboard SummaryWriter with log dir
    `WORK_DIR/LOGS_FOLDER/PROJECT`."""
    return auto_get_project(proj).start_pytorch_tensorboard(**kwargs)
