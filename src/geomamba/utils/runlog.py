"""Self-contained run directories: config snapshot + JSONL metrics + ckpts."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import yaml


class RunDir:
    def __init__(self, root: str | Path, name: str):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = Path(root) / f"{name}-{stamp}"
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "figs").mkdir(exist_ok=True)
        self._metrics = (self.path / "metrics.jsonl").open("a")

    def save_config(self, cfg: dict) -> None:
        (self.path / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    def log(self, **kv) -> None:
        kv.setdefault("time", time.time())
        self._metrics.write(json.dumps(kv) + "\n")
        self._metrics.flush()

    def save_ckpt(self, name: str, model: torch.nn.Module, **extra) -> None:
        torch.save({"model": model.state_dict(), **extra}, self.path / f"{name}.pt")
