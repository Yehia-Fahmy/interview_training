"""
Minimal starter for the App Deletion Risk Prediction challenge.

You are expected to implement the entire pipeline yourself. This file only
defines the high-level scaffolding so you have consistent entry points for
experiments.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


@dataclass
class ExperimentConfig:
    """Holds the knobs you'll likely need while iterating."""

    data_path: Path = Path("data.csv")
    seed: int = 42
    batch_size: int = 256
    max_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="App deletion risk training entry point.")
    parser.add_argument("--data-path", type=Path, default=ExperimentConfig.data_path)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=ExperimentConfig.max_epochs)
    parser.add_argument("--lr", type=float, default=ExperimentConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=ExperimentConfig.weight_decay)
    args = parser.parse_args()
    return ExperimentConfig(
        data_path=args.data_path,
        seed=args.seed,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )


def load_snapshot_table(path: Path) -> Any:
    """
    TODO: load `data.csv`, run sanity checks, and return the raw table you want to
    operate on (e.g., pandas DataFrame, Arrow table, polars LazyFrame, etc.).
    """
    raise NotImplementedError


def train_val_test_split(raw_table: Any, seed: int) -> Tuple[Any, Any, Any]:
    """
    TODO: implement group-aware splitting (users must not span splits) and return
    the three partitions you will feed into your feature pipeline.
    """
    raise NotImplementedError


def build_datasets(
    train_partition: Any, val_partition: Any, test_partition: Any
) -> Dict[str, torch.utils.data.Dataset]:
    """
    TODO: create PyTorch (or Lightning) Dataset objects that encapsulate feature
    engineering / normalization / augmentation logic.
    """
    raise NotImplementedError


class DeletionModel(torch.nn.Module):
    """
    TODO: define your architecture here. Feel free to extend this class or swap in
    LightningModules / Hydra configs—this class is just a placeholder so the file
    reminds you to keep model code separate from training orchestration.
    """

    def __init__(self) -> None:
        super().__init__()
        # define layers

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def train_loop(
    model: DeletionModel,
    datasets: Dict[str, torch.utils.data.Dataset],
    config: ExperimentConfig,
) -> Dict[str, float]:
    """
    TODO: implement everything needed for training:
      * DataLoader creation
      * Loss / metrics configuration
      * Optimization loop (or Lightning Trainer)
      * Checkpointing + logging
    Return any metrics you want to log for monitoring.
    """
    raise NotImplementedError


def main() -> None:
    config = parse_args()
    print(f"Starting experiment with config: {config}")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    raw_table = load_snapshot_table(config.data_path)
    train_part, val_part, test_part = train_val_test_split(raw_table, config.seed)
    datasets = build_datasets(train_part, val_part, test_part)

    model = DeletionModel().to(config.device)
    metrics = train_loop(model, datasets, config)
    print(f"Finished training. Metrics: {metrics}")


if __name__ == "__main__":
    main()

