# sparkline

> Experimental terminal viewer for TensorBoard training metrics — no browser required.

```
╔══════════════════════════════════════╗
║           s p a r k l i n e         ║
║    TensorBoard in your terminal     ║
╚══════════════════════════════════════╝
```

Uses [plotext](https://github.com/piccolomo/plotext) for beautiful ASCII charts and [rich](https://github.com/Textualize/rich) for styled terminal output.

## Status

This project is currently **experimental** and may change frequently.

---

## Features

- 📊 **ASCII line charts** rendered directly in your terminal via `plotext`
- 🔵 **TensorBoard** — reads `events.out.tfevents.*` files (TF1, TF2, PyTorch, tensorboardX)
- 🔄 **Watch mode** — auto-refreshes every N seconds for live training monitoring
- 🎛️ **EMA smoothing** — optional exponential moving-average to denoise noisy curves
- 🏷️ **Glob tag filtering** — e.g. `--tags "train/*"`
- 📋 **Stats table** — min / max / last / step count per metric
- 🗂️ **Group by tag or run** — flexible chart layout

---

## Installation

Installation is currently supported **from source only**.

From source:

```bash
git clone https://github.com/example/sparkline
cd sparkline
pip install -e .
```

---

## Quick Start

### TensorBoard logs

```bash
# Plot all metrics from a log directory
sparkline ./runs/

# Filter to specific metrics (glob patterns supported)
sparkline ./runs/ --tags "train/*" --tags val/loss

# Apply smoothing and group by run
sparkline ./runs/ --smoothing 0.7 --group-by run

# Watch mode: refresh every 5 seconds
sparkline ./runs/ --watch 5
```

---

## CLI Reference

```
Usage: sparkline [OPTIONS] [LOG_DIR]

  Plot training metrics from TensorBoard event files directly in the
  terminal — no browser required.

Options:
  LOG_DIR                          TensorBoard log directory.

  -r, --run RUN                    Restrict to a run by name (repeatable).
  -t, --tags PATTERN               Metric glob filter (repeatable).

  -g, --group-by [tag|run]         One chart per tag (default) or per run.
  -s, --smoothing FLOAT            EMA smoothing 0.0–1.0.  [default: 0.0]
      --height INT                 Chart height in rows.  [default: 18]
      --width INT                  Chart width in chars (default: auto).
      --no-summary                 Hide per-metric stats table.

  -w, --watch SECONDS              Live refresh interval.
  -v, --verbose                    Debug logging.
  -h, --help                       Show this message and exit.
  --version                        Show version and exit.
```

---

## Python API

```python
from sparkline import TensorBoardReader, MetricPlotter

# TensorBoard
reader = TensorBoardReader("./runs/").load()

plotter = MetricPlotter(smoothing=0.6, height=20)
plotter.plot_reader(reader)

# Iterate raw data
for run, tag, series in reader.iter_series():
    print(f"{run}/{tag}: {len(series)} steps, last={series.last:.4f}")
```

---

## Demo

```bash
python demo.py
```

Generates synthetic training curves for three optimisers and plots them.

---

## How it works

| Component | Responsibility |
|---|---|
| `readers/tensorboard.py` | Parses `events.out.tfevents.*` via TensorBoard's `EventFileLoader` |
| `utils.py` | `MetricSeries` dataclass — stores step / value / wall-time arrays |
| `plotter.py` | Renders ASCII charts with **plotext**, stats tables with **rich** |
| `cli.py` | Click-based CLI with watch mode and full option parsing |

---

## License

MIT
