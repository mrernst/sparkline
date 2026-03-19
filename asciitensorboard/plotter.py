"""
Terminal plotting with plotext.

Renders metric series as ASCII line charts directly in the terminal.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from asciitensorboard.utils import MetricSeries, abbreviate, format_value

# Plotext colour palette cycled for multiple runs / tags
_COLOURS = [
    "cyan", "green", "red", "yellow", "magenta", "blue",
    "bright_cyan", "bright_green", "bright_red",
]


def _get_rich():
    """Lazy-import rich so the package can be imported without it installed."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from rich import box
        return Console, Panel, Table, Text, box
    except ImportError as exc:
        raise ImportError(
            "rich is required for terminal display.  Install with: pip install rich"
        ) from exc


class MetricPlotter:
    """
    Plots MetricSeries data to the terminal using plotext + rich.

    Parameters
    ----------
    width:
        Desired plot width in characters (None = auto from terminal).
    height:
        Desired plot height in characters.
    smoothing:
        Exponential moving-average smoothing factor (0 = off, 0.9 = heavy).
    max_cols:
        Maximum number of plots side-by-side.
    theme:
        plotext colour theme: ``"default"`` or ``"dark"``.
    show_summary:
        Whether to print a stats table below each chart.
    """

    def __init__(
        self,
        width: Optional[int] = None,
        height: int = 18,
        smoothing: float = 0.0,
        max_cols: int = 2,
        theme: str = "default",
        show_summary: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.smoothing = smoothing
        self.max_cols = max_cols
        self.theme = theme
        self.show_summary = show_summary
        self._console = None  # lazy initialised on first use

    @property
    def console(self):
        if self._console is None:
            Console, *_ = _get_rich()
            self._console = Console()
        return self._console

    # ------------------------------------------------------------------
    # High-level entry points
    # ------------------------------------------------------------------

    def plot_reader(
        self,
        reader,
        tags: Optional[List[str]] = None,
        group_by: str = "tag",  # "tag" or "run"
    ) -> None:
        """
        Plot all metrics from a reader (TensorBoardReader or WandbReader).

        Parameters
        ----------
        reader:
            A loaded reader object.
        tags:
            Subset of tags to plot.  None = plot all.
        group_by:
            ``"tag"``  → one chart per metric, all runs overlaid.
            ``"run"``  → one chart per run, all metrics overlaid.
        """
        data = reader.get_data()  # {run: {tag: MetricSeries}}
        if not data:
            self.console.print("[yellow]No data found.[/yellow]")
            return

        runs = sorted(data.keys())
        all_tags = sorted({t for rd in data.values() for t in rd.keys()})
        if tags:
            import fnmatch
            all_tags = [t for t in all_tags if any(fnmatch.fnmatch(t, p) for p in tags)]

        if group_by == "tag":
            self._plot_grouped_by_tag(data, runs, all_tags)
        else:
            self._plot_grouped_by_run(data, runs, all_tags)

    def plot_series(
        self,
        series_map: Dict[str, MetricSeries],
        title: str = "",
    ) -> None:
        """
        Plot a ``{label: MetricSeries}`` dict as a single overlaid chart.
        """
        self._render_chart(series_map, title=title)

    # ------------------------------------------------------------------
    # Grouping strategies
    # ------------------------------------------------------------------

    def _plot_grouped_by_tag(
        self,
        data: Dict[str, Dict[str, MetricSeries]],
        runs: List[str],
        tags: List[str],
    ) -> None:
        """One chart per tag; all runs overlaid on the same chart."""
        self._print_header(data, runs)

        for tag in tags:
            series_map: Dict[str, MetricSeries] = {}
            for run in runs:
                s = data[run].get(tag)
                if s and len(s) > 0:
                    label = run if len(runs) > 1 else tag
                    series_map[label] = s
            if series_map:
                self._render_chart(series_map, title=tag)

    def _plot_grouped_by_run(
        self,
        data: Dict[str, Dict[str, MetricSeries]],
        runs: List[str],
        tags: List[str],
    ) -> None:
        """One chart per run; all tags overlaid on the same chart."""
        self._print_header(data, runs)

        for run in runs:
            series_map: Dict[str, MetricSeries] = {}
            for tag in tags:
                s = data[run].get(tag)
                if s and len(s) > 0:
                    series_map[tag] = s
            if series_map:
                self._render_chart(series_map, title=run)

    # ------------------------------------------------------------------
    # Core chart renderer
    # ------------------------------------------------------------------

    def _render_chart(
        self,
        series_map: Dict[str, MetricSeries],
        title: str = "",
    ) -> None:
        try:
            import plotext as plt
        except ImportError as exc:
            raise ImportError(
                "plotext is required for terminal plotting. "
                "Install it with: pip install plotext"
            ) from exc

        term_width = self.console.width or 100
        plot_width = min(self.width or term_width - 4, term_width - 4)
        plot_height = self.height

        plt.clf()
        plt.plotsize(plot_width, plot_height)

        if self.theme == "dark":
            plt.dark_background()

        plt.title(abbreviate(title, max_len=60))
        plt.xlabel("step")
        plt.ylabel("value")

        colour_idx = 0
        summary_rows: List[Tuple[str, MetricSeries]] = []

        for label, series in series_map.items():
            s = series.sorted()
            if not s:
                continue

            xs = s.steps
            ys = self._smooth(s.values)

            colour = _COLOURS[colour_idx % len(_COLOURS)]
            colour_idx += 1

            plt.plot(xs, ys, label=abbreviate(label, 30), color=colour)
            summary_rows.append((label, s))

        plt.theme("default")
        plt.show()

        if self.show_summary and summary_rows:
            self._print_summary(summary_rows)

        self.console.print()

    # ------------------------------------------------------------------
    # Stats summary table
    # ------------------------------------------------------------------

    def _print_summary(self, rows: List[Tuple[str, MetricSeries]]) -> None:
        _, _, Table, _, box = _get_rich()
        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold dim",
            padding=(0, 1),
        )
        table.add_column("Series", style="cyan", no_wrap=True)
        table.add_column("Steps", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Last", justify="right", style="bold")

        for label, s in rows:
            table.add_row(
                abbreviate(label, 35),
                str(len(s)),
                format_value(s.min),
                format_value(s.max),
                format_value(s.last),
            )

        self.console.print(table)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _print_header(
        self,
        data: Dict[str, Dict[str, MetricSeries]],
        runs: List[str],
    ) -> None:
        _, Panel, _, Text, _ = _get_rich()
        n_runs = len(runs)
        n_tags = len({t for rd in data.values() for t in rd})
        subtitle = (
            f"[dim]{n_runs} run{'s' if n_runs != 1 else ''}  ·  "
            f"{n_tags} metric{'s' if n_tags != 1 else ''}[/dim]"
        )
        self.console.print(
            Panel(
                Text.from_markup(subtitle, justify="center"),
                title="[bold]asciitensorboard[/bold]",
                border_style="bright_blue",
            )
        )
        self.console.print()

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    def _smooth(self, values: List[float]) -> List[float]:
        alpha = self.smoothing
        if alpha <= 0 or len(values) < 2:
            return values
        smoothed = [values[0]]
        for v in values[1:]:
            smoothed.append(alpha * smoothed[-1] + (1 - alpha) * v)
        return smoothed
