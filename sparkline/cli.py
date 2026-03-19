"""
CLI entry point for sparkline.

Usage examples
--------------
  # TensorBoard
  sparkline ./runs/
  sparkline ./runs/ --tags loss acc --smoothing 0.6
  sparkline ./runs/ --run my-experiment --group-by run

  # W&B (API)
  sparkline --wandb my-team/my-project
  sparkline --wandb my-team/my-project --max-runs 5 --tags train/loss

  # W&B (local offline)
  sparkline --wandb ./wandb/

  # Watch mode: refresh every N seconds
  sparkline ./runs/ --watch 5
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import click
from sparkline.plotter import MetricPlotter
from sparkline.utils import setup_logging

try:
    from rich.console import Console
    console = Console()
except ImportError:
    import re as _re
    class _FallbackConsole:
        width = 100
        def print(self, *a, **k):
            print(_re.sub(r'\[/?[^\]]+\]', '', ' '.join(str(x) for x in a)))
        def clear(self):
            import os; os.system('clear')
    console = _FallbackConsole()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HELP_EPILOG = """
\b
Examples:
  # Read a TensorBoard log directory
  sparkline ./runs/

  # Filter to specific metrics (glob patterns OK)
  sparkline ./runs/ --tags "train/*" --tags val/loss

  # Apply EMA smoothing (0.0–1.0, higher = smoother)
  sparkline ./runs/ --smoothing 0.8

  # Group by run instead of by metric
  sparkline ./runs/ --group-by run

  # Fetch from W&B cloud
  sparkline --wandb my-entity/my-project --max-runs 5

  # Read local W&B offline directory
  sparkline --wandb ./wandb/

  # Auto-refresh every 10 seconds
  sparkline ./runs/ --watch 10
"""


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


@click.command(
    name="sparkline",
    epilog=HELP_EPILOG,
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100},
)
@click.argument("log_dir", required=False, default=None, metavar="LOG_DIR")
# --- Source selection -------------------------------------------------------
@click.option(
    "--wandb", "wandb_source",
    default=None,
    metavar="ENTITY/PROJECT_OR_PATH",
    help="Load from W&B.  Pass 'entity/project' for the API, or a local wandb/ path for offline runs.",
)
# --- Filtering --------------------------------------------------------------
@click.option(
    "--run", "-r", "runs",
    multiple=True,
    metavar="RUN",
    help="Restrict to one or more runs by name (may be repeated).",
)
@click.option(
    "--tags", "-t", "tags",
    multiple=True,
    metavar="PATTERN",
    help="Only plot metrics matching this glob pattern (may be repeated).",
)
# --- Display ----------------------------------------------------------------
@click.option(
    "--group-by", "-g",
    type=click.Choice(["tag", "run"], case_sensitive=False),
    default="tag",
    show_default=True,
    help="Group charts by metric tag or by run.",
)
@click.option(
    "--smoothing", "-s",
    type=click.FloatRange(0.0, 1.0),
    default=0.0,
    show_default=True,
    help="EMA smoothing factor (0 = none, 0.9 = heavy).",
)
@click.option(
    "--height",
    type=int,
    default=18,
    show_default=True,
    help="Plot height in terminal rows.",
)
@click.option(
    "--width",
    type=int,
    default=None,
    help="Plot width in characters (default: auto).",
)
@click.option(
    "--no-summary",
    is_flag=True,
    default=False,
    help="Hide the per-metric stats table.",
)
# --- W&B specific -----------------------------------------------------------
@click.option(
    "--max-runs",
    type=int,
    default=10,
    show_default=True,
    help="Maximum number of W&B runs to fetch (most recent first).",
)
# --- Watch / live -----------------------------------------------------------
@click.option(
    "--watch", "-w",
    type=float,
    default=None,
    metavar="SECONDS",
    help="Re-read and re-plot every N seconds (live / watch mode).",
)
# --- Misc -------------------------------------------------------------------
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.version_option(package_name="sparkline", prog_name="sparkline")
def main(
    log_dir: Optional[str],
    wandb_source: Optional[str],
    runs: Tuple[str, ...],
    tags: Tuple[str, ...],
    group_by: str,
    smoothing: float,
    height: int,
    width: Optional[int],
    no_summary: bool,
    max_runs: int,
    watch: Optional[float],
    verbose: bool,
) -> None:
    """
    \b
    ╔══════════════════════════════════╗
    ║               s p a r k l i n e              ║
    ║  TensorBoard & W&B in your terminal  ║
    ╚══════════════════════════════════╝

    Plot training metrics from TensorBoard event files or Weights & Biases
    directly in the terminal — no browser required.
    """
    setup_logging(verbose)

    # Validate source
    if log_dir is None and wandb_source is None:
        raise click.UsageError(
            "Provide a LOG_DIR (TensorBoard) or --wandb ENTITY/PROJECT.",
        )
    if log_dir is not None and wandb_source is not None:
        raise click.UsageError("Use either LOG_DIR or --wandb, not both.")

    plotter = MetricPlotter(
        width=width,
        height=height,
        smoothing=smoothing,
        show_summary=not no_summary,
    )

    tag_list = list(tags) or None
    run_list = list(runs) or None

    def _do_plot() -> None:
        if wandb_source is not None:
            from sparkline.readers.wandb import WandbReader
            reader = WandbReader(
                source=wandb_source,
                run_ids=run_list,
                tags=tag_list,
                max_runs=max_runs,
            )
            try:
                reader.load()
            except (ImportError, RuntimeError) as exc:
                console.print(f"[red]Error:[/red] {exc}")
                sys.exit(1)
        else:
            from sparkline.readers.tensorboard import TensorBoardReader
            try:
                reader = TensorBoardReader(
                    log_dir=log_dir,
                    runs=run_list,
                    tags=tag_list,
                )
                reader.load()
            except FileNotFoundError as exc:
                console.print(f"[red]Error:[/red] {exc}")
                sys.exit(1)
            except ImportError as exc:
                console.print(f"[red]Error:[/red] {exc}")
                sys.exit(1)

        if not reader.runs_found:
            console.print(
                f"[yellow]No runs found.[/yellow] "
                f"Check that the path contains event files."
            )
            return

        plotter.plot_reader(reader, tags=tag_list, group_by=group_by)

    if watch is not None:
        _watch_loop(_do_plot, interval=watch)
    else:
        _do_plot()


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------


def _watch_loop(fn, interval: float) -> None:
    """Call *fn* repeatedly, clearing the screen between iterations."""
    try:
        while True:
            console.clear()
            fn()
            console.print(
                f"[dim]Refreshing every {interval:.0f}s — press Ctrl-C to quit[/dim]"
            )
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[dim]Watch mode stopped.[/dim]")


if __name__ == "__main__":
    main()
