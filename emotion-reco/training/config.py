"""Hyperparameters from Köksal & Gumus 2025, Section 3.1 / 3.4."""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    num_keyframes: int = 5           # → time dimension = N - 1 = 4
    convlstm_filters: int = 8
    convlstm_kernel: int = 1
    dense1: int = 2048
    dense2: int = 1024
    batch_size: int = 32
    epochs: int = 200
    learning_rate: float = 1e-3      # Adam defaults
    folds: int = 5
    random_state: int = 42
