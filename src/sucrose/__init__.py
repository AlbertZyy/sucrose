"""
Sucrose
=======

The sucrose package manages files of pytorch experiment projects.
"""

__all__ = [
    "scenario"
]

from .project import *
from .sucrose_logger import logger


def scenario(work_dir: str, name: str):
    """Start a Scenario."""
    from .project.scenario import set_current

    sc = Scenario(work_dir, name)
    set_current('Scenario', sc)

    return sc
