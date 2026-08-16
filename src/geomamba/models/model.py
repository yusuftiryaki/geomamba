"""Full model: embedding -> stacked blocks -> task heads.

Forecast head predicts the next-step delta (x_{t+1} - x_t) from h_t; the
anomaly head (M2+) scores each step from h_t and is ensembled with the
forecast residual at evaluation time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from geomamba.models.blocks import MambaBlock


@dataclass
class ModelConfig:
    variant: str = "mamba"  # mamba | mv_mamba | rotor_mamba | full_geo
    n_features: int = 4
    d_model: int = 128
    n_layers: int = 3
    expand: int = 2
    d_state: int = 16
    d_conv: int = 4
    scan_mode: str = "chunked"
    heads: list[str] = field(default_factory=lambda: ["forecast"])


_VARIANT_TRANSITIONS = {"mamba": None, "mv_mamba": "mv", "rotor_mamba": "rotor", "mv8_mamba": "mv8"}


def _make_block(cfg: ModelConfig) -> nn.Module:
    if cfg.variant not in _VARIANT_TRANSITIONS:
        raise ValueError(f"unknown variant: {cfg.variant}")
    return MambaBlock(
        cfg.d_model,
        expand=cfg.expand,
        d_conv=cfg.d_conv,
        d_state=cfg.d_state,
        scan_mode=cfg.scan_mode,
        transition=_VARIANT_TRANSITIONS[cfg.variant],
    )


class Backbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed = nn.Linear(cfg.n_features, cfg.d_model)
        self.norms = nn.ModuleList(
            nn.RMSNorm(cfg.d_model) for _ in range(cfg.n_layers)
        )
        self.blocks = nn.ModuleList(_make_block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.RMSNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, F] -> h: [B, L, d_model]"""
        h = self.embed(x)
        for norm, block in zip(self.norms, self.blocks):
            h = h + block(norm(h))
        return self.final_norm(h)


class GeoMamba(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg)
        self.heads = nn.ModuleDict()
        if "forecast" in cfg.heads:
            self.heads["forecast"] = nn.Linear(cfg.d_model, cfg.n_features)
        if "anomaly" in cfg.heads:
            self.heads["anomaly"] = nn.Linear(cfg.d_model, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {name: head(h) for name, head in self.heads.items()}

    @torch.no_grad()
    def rollout(self, context: torch.Tensor, horizon: int) -> torch.Tensor:
        """Autoregressive rollout: context [B, L, F] -> predictions [B, horizon, F]."""
        x = context
        preds = []
        for _ in range(horizon):
            delta = self.forward(x)["forecast"][:, -1]
            nxt = x[:, -1] + delta
            preds.append(nxt)
            x = torch.cat([x, nxt.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)
