
__all__ = [
    "LogDataFrame",
    "plot_evolution"
]

from typing import overload
from enum import Enum
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from matplotlib.lines import Line2D

from .project import Scenario
from .sucrose_logger import logger


def load_tensorboard_scalars(log_dir: str):
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        raise ImportError(
            "tensorboard is required to load scalars from tensorboard logs."
        )

    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    return ea


def tensorboard_scalars_to_dataframe(event_acc, tag: str, run_name: str):
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        raise ImportError(
            "tensorboard is required to generate dataframe from tensorboard scalars."
        )

    assert isinstance(event_acc, event_accumulator.EventAccumulator)
    try:
        events = event_acc.Scalars(tag)
    except KeyError:
        return None

    return pd.DataFrame({
        "step": [e.step for e in events],
        "value": [e.value for e in events],
        "wall_time": [e.wall_time for e in events],
        "run": run_name,
        "tag": tag,
    }).set_index("step")


class LogContext(Enum):
    TENSORBOARD = "tensorboard"


class LogDataFrame:
    """Log loader for scenarios with the given tags."""
    @overload
    def __init__(self, work_dir: str, *, runs: list[str], tags: list[str], meta_domain: str = "workspace"): ...
    @overload
    def __init__(self, *, runs: list[Scenario], tags: list[str], meta_domain: str = "workspace"): ...
    def __init__(self, work_dir=None, *, runs: list, tags: list[str], meta_domain: str = "workspace"):
        """
        Log loader for scenarios with the given tags.

        Args:
            work_dir (str, optional): The directory where the scenarios are located.
            runs (list[str] | list[Scenario]): The names of the scenarios to load.
                A list of Scenario instances can be used when work_dir is not specified.
            tags (list[str]): The tags of the scenarios to load.
        """
        if work_dir is None:
            if not all(isinstance(r, Scenario) for r in runs):
                raise ValueError("runs must be a list of Scenario instances "
                                 "when work_dir is not specified")
            self._scenarios = runs
        else:
            if not all(isinstance(r, str) for r in runs):
                raise ValueError("runs must be a list of strings "
                                 "when work_dir is specified")
            kwargs = {"meta_domain": meta_domain}
            self._scenarios = [Scenario(work_dir, r, **kwargs) for r in runs]

        self._tags = tags

    def load(self) -> DataFrame:
        frames: list[DataFrame] = []

        for sc in self._scenarios:
            ea = load_tensorboard_scalars(sc.LOGS_DIR)
            for tag in self._tags:
                df = tensorboard_scalars_to_dataframe(ea, tag, sc.NAME)
                if df is None:
                    logger.warning(f"No tag named {tag!r} in {sc.NAME!r}")
                    continue
                frames.append(df)

        return pd.concat(frames)


def plot_evolution(
    df: DataFrame,
    axes: Axes | None = None,
    fmts: dict[tuple[str, str], str] | list[str] | None = None,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    log_scale: str = "auto",
    legend_fmt: str = "{run} - {tag}",
) -> list[Line2D]:
    lines: list[Line2D] = []
    cursor = 0

    if axes is None:
        axes = plt.gca()

    for run, g in df.groupby("run"):
        for tag, gg in g.groupby("tag"):
            if isinstance(fmts, list) and cursor < len(fmts):
                args = gg.index, gg["value"], fmts[cursor]
            elif isinstance(fmts, dict) and (run, tag) in fmts:
                args = gg.index, gg["value"], fmts[(run, tag)]
            else:
                args = gg.index, gg["value"]

            line = axes.plot(*args, label=legend_fmt.format(run=run, tag=tag))
            lines.append(line)
            cursor += 1

    xlabel = df.index.name if xlabel is None else xlabel
    ylabel = "value" if ylabel is None else ylabel
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    if log_scale in ("auto", "AUTO", "Auto"):
        pass # TODO: decide log scale automatically
    else:
        if "y" in log_scale or "Y" in log_scale:
            axes.set_yscale("log")
        if "x" in log_scale or "X" in log_scale:
            axes.set_xscale("log")

    return lines
