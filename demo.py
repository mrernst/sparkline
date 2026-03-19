"""
demo.py — generate synthetic TensorBoard logs and plot them with asciitensorboard.

Run with:
    python demo.py
"""

import math
import os
import random
import sys
import tempfile
from pathlib import Path

# ── generate synthetic event files ──────────────────────────────────────────

def write_fake_events(log_dir: Path) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            print("Neither torch nor tensorboardX found; writing raw events via TF.")
            _write_via_tf(log_dir)
            return

    runs = {
        "run_Adam":   dict(lr=1e-3, noise=0.05, offset=0.0),
        "run_SGD":    dict(lr=5e-4, noise=0.10, offset=0.05),
        "run_RMSProp":dict(lr=2e-3, noise=0.07, offset=-0.02),
    }

    for name, cfg in runs.items():
        writer = SummaryWriter(log_dir=str(log_dir / name))
        for step in range(200):
            decay = math.exp(-cfg["lr"] * step * 3)
            train_loss = 2.5 * decay + cfg["offset"] + random.gauss(0, cfg["noise"])
            val_loss   = 2.5 * decay + cfg["offset"] + 0.1 + random.gauss(0, cfg["noise"] * 1.5)
            train_acc  = 1 - 0.9 * decay + random.gauss(0, cfg["noise"] * 0.3)
            val_acc    = 1 - 0.9 * decay - 0.05 + random.gauss(0, cfg["noise"] * 0.4)
            lr_val     = cfg["lr"] * (0.95 ** (step // 20))

            writer.add_scalar("train/loss",     max(train_loss, 0.01), step)
            writer.add_scalar("val/loss",       max(val_loss,   0.01), step)
            writer.add_scalar("train/accuracy", min(max(train_acc, 0.0), 1.0), step)
            writer.add_scalar("val/accuracy",   min(max(val_acc,   0.0), 1.0), step)
            writer.add_scalar("lr",             lr_val, step)
        writer.close()
        print(f"  wrote {name}")


def _write_via_tf(log_dir: Path) -> None:
    import tensorflow as tf

    runs = {
        "run_Adam":   dict(lr=1e-3, noise=0.05),
        "run_SGD":    dict(lr=5e-4, noise=0.10),
    }
    for name, cfg in runs.items():
        writer = tf.summary.create_file_writer(str(log_dir / name))
        for step in range(200):
            decay = math.exp(-cfg["lr"] * step * 3)
            with writer.as_default():
                tf.summary.scalar("train/loss",     max(2.5 * decay + random.gauss(0, cfg["noise"]), 0.01), step=step)
                tf.summary.scalar("val/loss",       max(2.5 * decay + 0.1 + random.gauss(0, cfg["noise"] * 1.5), 0.01), step=step)
                tf.summary.scalar("train/accuracy", min(max(1 - 0.9 * decay + random.gauss(0, 0.03), 0.0), 1.0), step=step)
        writer.flush()
        print(f"  wrote {name}")


# ── plot ─────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(42)

    with tempfile.TemporaryDirectory(prefix="asciitb_demo_") as tmpdir:
        log_dir = Path(tmpdir)
        print(f"\nGenerating synthetic TensorBoard logs in {log_dir} …\n")
        write_fake_events(log_dir)

        print("\n" + "═" * 60)
        print("  asciitensorboard demo — all metrics, grouped by tag")
        print("═" * 60 + "\n")

        from asciitensorboard.readers.tensorboard import TensorBoardReader
        from asciitensorboard.plotter import MetricPlotter

        reader = TensorBoardReader(log_dir=log_dir).load()
        plotter = MetricPlotter(smoothing=0.6, height=16, show_summary=True)
        plotter.plot_reader(reader, group_by="tag")

        print("\n" + "═" * 60)
        print("  Same data — grouped by run")
        print("═" * 60 + "\n")
        plotter.plot_reader(reader, group_by="run")
        
        
        print("\n" + "═" * 60)
        print("  asciitensorboard demo — all metrics, grouped by tag")
        print("═" * 60 + "\n")
        
        
        from asciitensorboard.readers.wandb import WandbReader
        log_dir = '/Users/markus/Desktop/mtrlx-sac-test_20260224_153230_seed42/wandb/run-20260224_153251-d729beb1-6a0b-463c-9b73-b2305cf43b0b'
        reader = WandbReader(source=log_dir).load()
        plotter = MetricPlotter(smoothing=0.6, height=16, show_summary=True)
        plotter.plot_reader(reader, group_by="tag")
        
        print("\n" + "═" * 60)
        print("  Same data — grouped by run")
        print("═" * 60 + "\n")
        plotter.plot_reader(reader, group_by="run")


if __name__ == "__main__":
    main()
