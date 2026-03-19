"""
sparkline — View TensorBoard & W&B metrics right in your terminal.
"""

__version__ = "0.1.0"
__author__ = "sparkline contributors"

from sparkline.plotter import MetricPlotter
from sparkline.readers.tensorboard import TensorBoardReader
from sparkline.readers.wandb import WandbReader

__all__ = ["MetricPlotter", "TensorBoardReader", "WandbReader"]
