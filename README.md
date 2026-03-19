# asciitensorboard

> View TensorBoard & Weights and Biases training metrics **directly in your terminal** — no browser required.

```
╔══════════════════════════════════════╗
║   a s c i i t e n s o r b o a r d  ║
║  TensorBoard & W&B in your terminal ║
╚══════════════════════════════════════╝
```

Uses [plotext](https://github.com/piccolomo/plotext) for beautiful ASCII charts and [rich](https://github.com/Textualize/rich) for styled terminal output.

---

## Features

- 📊 **ASCII line charts** rendered directly in your terminal via `plotext`
- 🔵 **TensorBoard** — reads `events.out.tfevents.*` files (TF1, TF2, PyTorch, tensorboardX)
- 🟡 **W&B** — reads from the W&B cloud API *or* local offline run directories
- 🔄 **Watch mode** — auto-refreshes every N seconds for live training monitoring
- 🎛️ **EMA smoothing** — optional exponential moving-average to denoise noisy curves
- 🏷️ **Glob tag filtering** — e.g. `--tags "train/*"`
- 📋 **Stats table** — min / max / last / step count per metric
- 🗂️ **Group by tag or run** — flexible chart layout

---

## Installation

```bash
pip install asciitensorboard
```

With W&B support:

```bash
pip install "asciitensorboard[wandb]"
```

From source:

```bash
git clone https://github.com/example/asciitensorboard
cd asciitensorboard
pip install -e ".[all]"
```

---

## Quick Start

### TensorBoard logs

```bash
# Plot all metrics from a log directory
asciitb ./runs/

# Filter to specific metrics (glob patterns supported)
asciitb ./runs/ --tags "train/*" --tags val/loss

# Apply smoothing and group by run
asciitb ./runs/ --smoothing 0.7 --group-by run

# Watch mode: refresh every 5 seconds
asciitb ./runs/ --watch 5
```

### Weights & Biases

```bash
# Fetch from the W&B cloud API
export WANDB_API_KEY=your_key_here
asciitb --wandb my-entity/my-project

# Limit runs and filter metrics
asciitb --wandb my-entity/my-project --max-runs 3 --tags accuracy loss

# Read a local offline W&B directory (no API key needed)
asciitb --wandb ./wandb/
```

---

## CLI Reference

```
Usage: asciitb [OPTIONS] [LOG_DIR]

  Plot training metrics from TensorBoard event files or Weights & Biases
  directly in the terminal — no browser required.

Options:
  LOG_DIR                          TensorBoard log directory.
  --wandb ENTITY/PROJECT_OR_PATH   Load from W&B (API or local path).

  -r, --run RUN                    Restrict to a run by name (repeatable).
  -t, --tags PATTERN               Metric glob filter (repeatable).

  -g, --group-by [tag|run]         One chart per tag (default) or per run.
  -s, --smoothing FLOAT            EMA smoothing 0.0–1.0.  [default: 0.0]
      --height INT                 Chart height in rows.  [default: 18]
      --width INT                  Chart width in chars (default: auto).
      --no-summary                 Hide per-metric stats table.

      --max-runs INT               Max W&B runs to fetch.  [default: 10]

  -w, --watch SECONDS              Live refresh interval.
  -v, --verbose                    Debug logging.
  -h, --help                       Show this message and exit.
  --version                        Show version and exit.
```

---

## Python API

```python
from asciitensorboard import TensorBoardReader, WandbReader, MetricPlotter

# TensorBoard
reader = TensorBoardReader("./runs/").load()

plotter = MetricPlotter(smoothing=0.6, height=20)
plotter.plot_reader(reader)

# W&B (API)
reader = WandbReader("my-team/my-project", max_runs=5).load()
plotter.plot_reader(reader, tags=["train/loss", "val/loss"])

# W&B (local)
reader = WandbReader("./wandb/").load()
plotter.plot_reader(reader, group_by="run")

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
| `readers/wandb.py` | Fetches history via the W&B API or parses local `wandb-history.jsonl` |
| `utils.py` | `MetricSeries` dataclass — stores step / value / wall-time arrays |
| `plotter.py` | Renders ASCII charts with **plotext**, stats tables with **rich** |
| `cli.py` | Click-based CLI with watch mode and full option parsing |

---

## License

MIT
