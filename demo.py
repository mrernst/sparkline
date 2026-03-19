"""
demo.py — generate synthetic TensorBoard logs and plot them with sparkline.

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


def write_fake_wandb_run(wandb_dir: Path) -> None:
    """Generate a synthetic W&B run locally (offline mode)."""
    try:
        import wandb
    except ImportError:
        print("  (wandb not installed; skipping W&B demo — install with: pip install wandb)")
        return

    wandb_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize a W&B run in offline mode
    with wandb.init(
        project="sparkline-demo",
        entity="demo-user",
        dir=str(wandb_dir),
        mode="offline",
    ):
        configs = {
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "batch_size": 32,
        }
        wandb.config.update(configs)

        for step in range(200):
            decay = math.exp(-1e-3 * step * 3)
            train_loss = 2.5 * decay + random.gauss(0, 0.05)
            val_loss   = 2.5 * decay + 0.1 + random.gauss(0, 0.08)
            train_acc  = 1 - 0.9 * decay + random.gauss(0, 0.015)
            val_acc    = 1 - 0.9 * decay - 0.05 + random.gauss(0, 0.02)
            lr_val     = 1e-3 * (0.95 ** (step // 20))

            wandb.log({
                "train/loss":     max(train_loss, 0.01),
                "val/loss":       max(val_loss,   0.01),
                "train/accuracy": min(max(train_acc, 0.0), 1.0),
                "val/accuracy":   min(max(val_acc,   0.0), 1.0),
                "learning_rate":  lr_val,
            }, step=step)

    print("  wrote wandb run in offline mode")


# ── plot ─────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(42)

    with tempfile.TemporaryDirectory(prefix="sparkline_demo_") as tmpdir:
        log_dir = Path(tmpdir)
        print(f"\nGenerating synthetic TensorBoard logs in {log_dir} …\n")
        write_fake_events(log_dir)

        print("\n" + "═" * 60)
        print("  sparkline demo — TensorBoard: grouped by tag")
        print("═" * 60 + "\n")

        from sparkline.readers.tensorboard import TensorBoardReader
        from sparkline.readers.wandb import WandbReader
        from sparkline.plotter import MetricPlotter

        reader = TensorBoardReader(log_dir=log_dir).load()
        plotter = MetricPlotter(smoothing=0.6, height=16, show_summary=True)
        plotter.plot_reader(reader, group_by="tag")

        print("\n" + "═" * 60)
        print("  TensorBoard: Same data — grouped by run")
        print("═" * 60 + "\n")
        plotter.plot_reader(reader, group_by="run")

        # ── W&B Demo ────────────────────────────────────────────────────────

        wandb_dir = Path(tmpdir) / "wandb_demo"
        print("\n" + "═" * 60)
        print("  Generating synthetic W&B logs in offline mode…")
        print("═" * 60 + "\n")
        write_fake_wandb_run(wandb_dir)

        # Try to load and plot W&B data if available
        if (wandb_dir / "wandb").exists():
            print("\n" + "═" * 60)
            print("  sparkline demo — W&B (offline): grouped by tag")
            print("═" * 60 + "\n")
            
            try:
                reader_wandb = WandbReader(source=str(wandb_dir / "wandb")).load()
                if reader_wandb.runs_found:
                    plotter.plot_reader(reader_wandb, group_by="tag")

                    print("\n" + "═" * 60)
                    print("  W&B: Same data — grouped by run")
                    print("═" * 60 + "\n")
                    plotter.plot_reader(reader_wandb, group_by="run")
                else:
                    print("[cyan]No W&B runs found.[/cyan]")
            except Exception as exc:
                print(f"[yellow]W&B demo skipped: {exc}[/yellow]")


if __name__ == "__main__":
    main()
