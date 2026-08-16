"""npz trajectory files -> normalized torch Datasets.

File layout (written by scripts/generate_data.py): arrays 'train', 'val',
'test' of shape [N, L, F]. Normalization stats come from the train split of
the FULL file; data-efficiency subsets (n_train) reuse the same stats so
curves differ only in what the model sees.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Yields (x_t sequence, next-step delta targets)."""

    def __init__(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray):
        x = (x - mean) / std
        self.x = torch.from_numpy(x.astype(np.float32))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        traj = self.x[i]
        return {"input": traj[:-1], "target": traj[1:] - traj[:-1]}


def load_splits(
    path: str | Path,
    n_train: int | None = None,
    seed: int = 0,
    max_len: int | None = None,
) -> dict[str, TrajectoryDataset]:
    data = np.load(path)
    train = data["train"]
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True) + 1e-8
    if n_train is not None and n_train < len(train):
        idx = np.random.default_rng(seed).permutation(len(train))[:n_train]
        train = train[idx]
    return {
        name: TrajectoryDataset(arr[:, :max_len], mean, std)
        for name, arr in [("train", train), ("val", data["val"]), ("test", data["test"])]
    }
