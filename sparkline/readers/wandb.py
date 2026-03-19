"""
Weights & Biases (W&B) log reader.

Supports two modes:

1. **API mode** (default) — uses the ``wandb`` Python SDK to pull runs from the
   W&B cloud (or a self-hosted server).  Requires WANDB_API_KEY to be set.

2. **Local / offline mode** — reads the binary ``run-<ID>.wandb`` protobuf file
   produced by every W&B run (online or offline) directly, without any network
   access.  This is the format used by all modern versions of the wandb SDK
   (≥ 0.12).  Very old runs that only have ``wandb-history.jsonl`` are also
   handled as a fallback.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

from sparkline.utils import MetricSeries, logger


class WandbReader:
    """
    Read scalar metrics from W&B runs.

    Parameters
    ----------
    source:
        One of:

        * A W&B entity/project string like ``"my-team/my-project"`` — loads
          runs from the W&B API (requires ``wandb`` installed + API key).
        * A local path to a ``wandb/`` directory — reads the binary
          ``run-*.wandb`` protobuf files directly, no network access needed.

    run_ids:
        Optionally filter to specific run IDs / names.
    tags:
        Optionally filter metrics by tag name (glob patterns supported).
    max_runs:
        Maximum number of runs to fetch from the API (most recent first).
    """

    def __init__(
        self,
        source: str,
        run_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        max_runs: int = 10,
    ) -> None:
        self.source = source
        self.run_ids = run_ids
        self.tags = tags
        self.max_runs = max_runs
        self._data: Dict[str, Dict[str, MetricSeries]] = {}

    # ------------------------------------------------------------------
    # Public API (mirrors TensorBoardReader)
    # ------------------------------------------------------------------

    def load(self) -> "WandbReader":
        """Load runs.  Returns *self* for chaining."""
        src = Path(self.source).expanduser()
        if src.exists() and src.is_dir():
            self._load_local(src)
        else:
            self._load_api(self.source)
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
        for run, tag_map in sorted(self._data.items()):
            for tag, series in sorted(tag_map.items()):
                yield run, tag, series

    def get_data(self) -> Dict[str, Dict[str, MetricSeries]]:
        return self._data

    # ------------------------------------------------------------------
    # Private: API mode
    # ------------------------------------------------------------------

    def _load_api(self, entity_project: str) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is required to fetch runs from the W&B API. "
                "Install it with: pip install wandb"
            ) from exc

        api = wandb.Api()
        logger.debug(f"Fetching runs from W&B: {entity_project}")
        try:
            runs = api.runs(entity_project)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch W&B runs for '{entity_project}': {exc}\n"
                "Make sure WANDB_API_KEY is set and the project path is correct."
            ) from exc

        count = 0
        for run in runs:
            if self.run_ids and run.id not in self.run_ids and run.name not in self.run_ids:
                continue
            run_label = f"{run.name} ({run.id})"
            logger.debug(f"Loading W&B run: {run_label}")
            self._data[run_label] = self._extract_api_run(run)
            count += 1
            if count >= self.max_runs:
                break

    def _extract_api_run(self, run) -> Dict[str, MetricSeries]:
        """Pull history from a wandb.apis.public.Run object."""
        series_map: Dict[str, MetricSeries] = {}

        try:
            history = run.history(samples=10_000, pandas=True)
        except Exception as exc:
            logger.warning(f"Could not fetch history for run {run.name}: {exc}")
            return series_map

        step_col = "_step" if "_step" in history.columns else None
        time_col = "_timestamp" if "_timestamp" in history.columns else None

        for col in history.columns:
            if col.startswith("_"):
                continue
            if not self._tag_allowed(col):
                continue
            try:
                import numpy as np
                col_data = history[col].dropna()
                if col_data.empty:
                    continue
                numeric = col_data.apply(lambda x: isinstance(x, (int, float, np.number)))
                col_data = col_data[numeric]
                if col_data.empty:
                    continue

                series = MetricSeries()
                for idx in col_data.index:
                    step = int(history.loc[idx, step_col]) if step_col else idx
                    wt = float(history.loc[idx, time_col]) if time_col else None
                    series.append(step=step, value=float(col_data.loc[idx]), wall_time=wt)
                series_map[col] = series
            except Exception as exc:
                logger.debug(f"Skipping column {col}: {exc}")

        return series_map

    # ------------------------------------------------------------------
    # Private: local mode — dispatch to the right parser
    # ------------------------------------------------------------------

    def _load_local(self, wandb_dir: Path) -> None:
        """Scan *wandb_dir* for run directories and parse each one."""
        run_dirs = self._find_run_dirs(wandb_dir)
        if not run_dirs:
            logger.warning(f"No W&B run directories found under {wandb_dir}")
            return

        for run_dir in run_dirs:
            run_name = run_dir.name
            if self.run_ids and not any(rid in run_name for rid in self.run_ids):
                continue
            logger.debug(f"Reading local W&B run: {run_name}")

            data = self._parse_run_dir(run_dir)
            if data:
                self._data[run_name] = data

    def _find_run_dirs(self, wandb_dir: Path) -> List[Path]:
        """
        Locate run directories.

        Modern W&B writes to ``run-TIMESTAMP-RUNID/`` and offline runs go to
        ``offline-run-TIMESTAMP-RUNID/``.  Each contains a ``run-<ID>.wandb``
        binary file at its root.
        """
        candidates = []
        for entry in sorted(wandb_dir.iterdir()):
            #if not entry.is_dir():
            #    continue
            #print(entry)
            # Heuristic: a run directory contains a .wandb binary OR an old jsonl
            has_wandb_bin = (entry.suffix == ".wandb")
            has_jsonl = (entry / "files" / "wandb-history.jsonl").exists() or \
                        (entry / "wandb-history.jsonl").exists()
            if has_wandb_bin or has_jsonl:
                candidates.append(entry)
        return candidates

    def _parse_run_dir(self, run_dir: Path) -> Dict[str, MetricSeries]:
        """Try the binary .wandb format first, fall back to legacy jsonl."""
        
        result = self._parse_wandb_binary(run_dir)
        return result
             
        wandb_files = sorted(run_dir.glob("run-*.wandb"))
        if wandb_files:
            for wf in wandb_files:
                # Skip the tiny .wandb.synced marker files
                if wf.suffix == ".synced":
                    continue
                result = self._parse_wandb_binary(wf)
                if result:
                    return result

        # Legacy fallback: wandb-history.jsonl (wandb < 0.12 / old offline dirs)
        return self._parse_legacy_jsonl(run_dir)

    # ------------------------------------------------------------------
    # Modern binary format: run-<ID>.wandb
    # ------------------------------------------------------------------

    def _parse_wandb_binary(self, path: Path) -> Dict[str, MetricSeries]:
        """
        Parse a ``run-<ID>.wandb`` binary protobuf file using wandb's own
        internal datastore reader.

        The file is a length-prefixed stream of ``wandb_internal_pb2.Record``
        messages.  We extract ``history`` records, each of which carries a list
        of ``HistoryItem`` protos whose ``value_json`` field contains a JSON-
        encoded scalar (or array/object for non-scalar types we skip).
        """
        try:
            from wandb.proto import wandb_internal_pb2
            from wandb.sdk.internal import datastore as wandb_datastore
        except ImportError as exc:
            raise ImportError(
                "wandb is required to read local .wandb files. "
                "Install it with: pip install wandb"
            ) from exc

        series_map: Dict[str, MetricSeries] = {}
        step_counter = 0  # monotonic fallback if the record has no explicit step

        ds = wandb_datastore.DataStore()
        try:
            ds.open_for_scan(str(path))
        except Exception as exc:
            logger.warning(f"Could not open {path}: {exc}")
            return series_map

        while True:
            try:
                raw = ds.scan_data()
            except AssertionError:
                # Checksum errors can happen in crashed/incomplete run files.
                # wandb's own sync tool handles this by stopping at the error.
                logger.debug(f"Checksum error in {path} — stopping scan here")
                break
            except Exception as exc:
                logger.debug(f"Scan error in {path}: {exc}")
                break

            if raw is None:
                break  # EOF

            pb = wandb_internal_pb2.Record()
            try:
                pb.ParseFromString(raw)
            except Exception as exc:
                logger.debug(f"Proto parse error: {exc}")
                continue

            record_type = pb.WhichOneof("record_type")

            if record_type == "history":
                # Each history record = one wandb.log() call.
                # Items carry key + value_json; some also carry nested_key for
                # structured metrics like {"train": {"loss": 0.5}}.
                explicit_step: Optional[int] = None

                for item in pb.history.item:
                    key = item.key
                    # Handle nested keys (e.g. item.nested_key = ["train", "loss"])
                    if not key and item.nested_key:
                        key = "/".join(item.nested_key)

                    if key == "_step":
                        try:
                            explicit_step = int(json.loads(item.value_json))
                        except (ValueError, json.JSONDecodeError):
                            pass
                        continue

                    if key.startswith("_"):
                        continue

                    if not self._tag_allowed(key):
                        continue

                    try:
                        value = json.loads(item.value_json)
                    except (ValueError, json.JSONDecodeError):
                        continue

                    if not isinstance(value, (int, float)):
                        continue  # skip arrays, dicts, strings, etc.

                    step = explicit_step if explicit_step is not None else step_counter
                    if key not in series_map:
                        series_map[key] = MetricSeries()
                    series_map[key].append(step=step, value=float(value))

                step_counter += 1

            elif record_type == "stats":
                # System metrics (GPU/CPU/mem) — skip unless the user filtered for them
                pass

        return series_map

    # ------------------------------------------------------------------
    # Legacy fallback: wandb-history.jsonl
    # ------------------------------------------------------------------

    def _parse_legacy_jsonl(self, run_dir: Path) -> Dict[str, MetricSeries]:
        candidates = [
            run_dir / "files" / "wandb-history.jsonl",
            run_dir / "wandb-history.jsonl",
        ]
        history_file = next((p for p in candidates if p.exists()), None)
        if history_file is None:
            return {}

        logger.debug(f"Falling back to legacy jsonl: {history_file}")
        series_map: Dict[str, MetricSeries] = {}

        with open(history_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                step = record.get("_step", 0)
                wall_time = record.get("_timestamp")

                for key, value in record.items():
                    if key.startswith("_"):
                        continue
                    if not isinstance(value, (int, float)):
                        continue
                    if not self._tag_allowed(key):
                        continue
                    if key not in series_map:
                        series_map[key] = MetricSeries()
                    series_map[key].append(step=step, value=float(value), wall_time=wall_time)

        return series_map

    # ------------------------------------------------------------------

    def _tag_allowed(self, tag: str) -> bool:
        if self.tags is None:
            return True
        import fnmatch
        return any(fnmatch.fnmatch(tag, pat) for pat in self.tags)
