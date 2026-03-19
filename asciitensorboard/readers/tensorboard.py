"""
TensorBoard event file reader.

Reads scalar summaries from TensorBoard event files produced by
TensorFlow, PyTorch (via torch.utils.tensorboard or tensorboardX), etc.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

from asciitensorboard.utils import MetricSeries, logger


class TensorBoardReader:
    """
    Reads scalar metrics from TensorBoard event files.

    Parameters
    ----------
    log_dir:
        Path to the root TensorBoard log directory (or a single run sub-dir).
        The reader will search recursively for ``events.out.tfevents.*`` files.
    runs:
        Optional list of run names (sub-directory names) to restrict to.
        If *None*, all discovered runs are loaded.
    tags:
        Optional list of tag patterns to restrict which metrics are loaded.
        Simple glob-style ``*`` wildcards are supported.
    """

    def __init__(
        self,
        log_dir: str | os.PathLike,
        runs: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.runs = runs
        self.tags = tags
        self._data: Dict[str, Dict[str, MetricSeries]] = {}  # run -> tag -> series

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "TensorBoardReader":
        """Discover and parse all matching event files.  Returns *self* for chaining."""
        event_files = self._discover_event_files()
        if not event_files:
            logger.warning(f"No TensorBoard event files found under {self.log_dir}")
            return self

        for run_name, path in event_files:
            logger.debug(f"Reading {path}  (run={run_name!r})")
            try:
                series_map = self._parse_event_file(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to parse {path}: {exc}")
                continue

            if run_name not in self._data:
                self._data[run_name] = {}
            for tag, series in series_map.items():
                if run_name in self._data and tag in self._data[run_name]:
                    # Merge (e.g. multiple event files for the same run)
                    self._data[run_name][tag].extend(series)
                else:
                    self._data[run_name][tag] = series

        return self

    @property
    def runs_found(self) -> List[str]:
        return sorted(self._data.keys())

    def tags_for_run(self, run: str) -> List[str]:
        return sorted(self._data.get(run, {}).keys())

    def all_tags(self) -> List[str]:
        tags: set[str] = set()
        for run_data in self._data.values():
            tags.update(run_data.keys())
        return sorted(tags)

    def get_series(self, run: str, tag: str) -> Optional[MetricSeries]:
        return self._data.get(run, {}).get(tag)

    def iter_series(self) -> Generator[Tuple[str, str, MetricSeries], None, None]:
        """Yield ``(run, tag, series)`` tuples for every loaded metric."""
        for run, tag_map in sorted(self._data.items()):
            for tag, series in sorted(tag_map.items()):
                yield run, tag, series

    def get_data(self) -> Dict[str, Dict[str, MetricSeries]]:
        """Return the raw ``{run: {tag: MetricSeries}}`` dict."""
        return self._data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_event_files(self) -> List[Tuple[str, Path]]:
        """Return ``[(run_name, path), ...]`` for every event file found."""
        results: List[Tuple[str, Path]] = []

        if not self.log_dir.exists():
            raise FileNotFoundError(f"Log directory does not exist: {self.log_dir}")

        # If log_dir itself contains event files, treat it as a single run
        direct_files = sorted(self.log_dir.glob("events.out.tfevents.*"))
        if direct_files:
            run_name = self.log_dir.name
            if self._run_allowed(run_name):
                for f in direct_files:
                    results.append((run_name, f))

        # Also recurse into sub-directories
        for sub in sorted(self.log_dir.iterdir()):
            if not sub.is_dir():
                continue
            run_name = sub.name
            if not self._run_allowed(run_name):
                continue
            for event_file in sorted(sub.rglob("events.out.tfevents.*")):
                results.append((run_name, event_file))

        return results

    def _run_allowed(self, name: str) -> bool:
        if self.runs is None:
            return True
        return name in self.runs

    def _tag_allowed(self, tag: str) -> bool:
        if self.tags is None:
            return True
        import fnmatch
        return any(fnmatch.fnmatch(tag, pat) for pat in self.tags)

    def _parse_event_file(self, path: Path) -> Dict[str, MetricSeries]:
        """Parse a single event file and return ``{tag: MetricSeries}``."""
        try:
            from tensorboard.backend.event_processing import event_file_loader
        except ImportError as exc:
            raise ImportError(
                "tensorboard is required to read TensorBoard logs. "
                "Install it with: pip install tensorboard"
            ) from exc

        series_map: Dict[str, MetricSeries] = defaultdict(MetricSeries)
        loader = event_file_loader.EventFileLoader(str(path))

        for event in loader.Load():
            if not event.HasField("summary"):
                continue
            step = event.step
            wall_time = event.wall_time

            for value in event.summary.value:
                tag = value.tag
                if not self._tag_allowed(tag):
                    continue

                # TF2-style SimpleValue
                scalar = None
                if value.HasField("simple_value"):
                    scalar = value.simple_value
                # TF2 tensor-based scalars
                elif value.HasField("tensor"):
                    try:
                        import numpy as np
                        from tensorboard.compat.tensorflow_stub import dtypes
                        tensor = value.tensor
                        scalar = tensor.float_val[0]
                        if tensor.dtype in (dtypes.float32, dtypes.float64,
                                            dtypes.int32, dtypes.int64):
                            arr = np.frombuffer(
                                tensor.tensor_content,
                                dtype=dtypes.as_numpy_dtype(tensor.dtype),
                            )
                            if arr.size == 1:
                                scalar = float(arr[0])
                        elif tensor.float_val:
                            scalar = tensor.float_val[0]
                        elif tensor.double_val:
                            scalar = tensor.double_val[0]
                        elif tensor.int_val:
                            scalar = tensor.int_val[0]
                    except Exception:
                        pass
                # Older metadata plugin approach
                elif value.metadata.plugin_data.plugin_name == "scalars":
                    pass  # already handled above

                if scalar is not None:
                    series_map[tag].append(step=step, value=scalar, wall_time=wall_time)

        return series_map
