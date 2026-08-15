"""Step-based trainer: AdamW + warmup/cosine, JSONL logging, best/last ckpt."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from geomamba.models.model import GeoMamba
from geomamba.train import metrics as M
from geomamba.utils.runlog import RunDir


@dataclass
class TrainConfig:
    steps: int = 30_000
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    eval_every: int = 2000
    log_every: int = 100
    num_threads: int = 3
    compile: bool = True
    rollout_context: int = 64
    rollout_horizons: tuple[int, ...] = (1, 10, 50)
    eval_rollout_batches: int = 2
    eval_tf_batches: int = 16
    max_wall_seconds: int = 2 * 3600


class Trainer:
    def __init__(self, cfg: TrainConfig, run: RunDir):
        self.cfg = cfg
        self.run = run

    def _lr_at(self, step: int) -> float:
        c = self.cfg
        if step < c.warmup_steps:
            return c.lr * (step + 1) / c.warmup_steps
        t = (step - c.warmup_steps) / max(1, self.cfg.steps - c.warmup_steps)
        return c.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    def fit(self, model: GeoMamba, train_ds, val_ds) -> dict:
        c = self.cfg
        torch.set_num_threads(c.num_threads)
        opt = torch.optim.AdamW(
            model.parameters(), lr=c.lr, weight_decay=c.weight_decay
        )
        train_loader = DataLoader(
            train_ds, batch_size=c.batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(val_ds, batch_size=c.batch_size)

        # compile the training forward only; eval (esp. autoregressive rollout
        # with growing sequence length) runs the eager model to avoid recompiles
        train_model = model
        if c.compile:
            try:
                train_model = torch.compile(model)
            except Exception as e:  # pragma: no cover - platform dependent
                self.run.log(event="compile_failed", error=repr(e))

        best_val = float("inf")
        step, t0 = 0, time.time()
        data_iter = iter(train_loader)
        model.train()
        while step < c.steps and (time.time() - t0) < c.max_wall_seconds:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            for group in opt.param_groups:
                group["lr"] = self._lr_at(step)
            pred = train_model(batch["input"])["forecast"]
            loss = torch.nn.functional.mse_loss(pred, batch["target"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
            opt.step()
            step += 1

            if step % c.log_every == 0:
                self.run.log(step=step, split="train", loss=loss.item(),
                             lr=opt.param_groups[0]["lr"])
            if step % c.eval_every == 0 or step == c.steps:
                model.eval()
                val_mse = M.teacher_forced_mse(model, val_loader, c.eval_tf_batches)
                roll = M.rollout_mse(
                    model, val_loader, c.rollout_context, tuple(c.rollout_horizons),
                    max_batches=c.eval_rollout_batches,
                )
                self.run.log(step=step, split="val", mse=val_mse, **roll)
                if val_mse < best_val:
                    best_val = val_mse
                    self.run.save_ckpt("best", model, step=step, val_mse=val_mse)
                model.train()

        model.eval()
        self.run.save_ckpt("last", model, step=step)
        persistence = M.persistence_mse(val_loader)
        self.run.log(step=step, split="final", best_val_mse=best_val,
                     persistence_mse=persistence,
                     wall_seconds=round(time.time() - t0, 1))
        return {"best_val_mse": best_val, "persistence_mse": persistence,
                "steps": step, "wall_seconds": time.time() - t0}
