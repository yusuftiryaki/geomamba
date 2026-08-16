"""Full model: embedding -> stacked blocks -> task heads.

Forecast head predicts the next-step delta (x_{t+1} - x_t) from h_t; the
anomaly head (M2+) scores each step from h_t and is ensembled with the
forecast residual at evaluation time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from geomamba.algebra import Algebra
from geomamba.models.blocks import MambaBlock
from geomamba.models.geo_layers import GeoEmbed, GeoHead, GradeRMSNorm


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
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    lambda_init: float = -3.0
    # representation ablation: slot inputs into blade positions of slot_grade
    # and keep channel mixing / normalisation grade-respecting
    geo_embed: bool = False
    geo_proj: bool = False
    slot_grade: int = 2  # rigid-body omega and B fields are bivectors
    # controls for "is the slotting win geometric?" (report/preregistration_slot_controls.md)
    slot_blades: list[int] | None = None  # explicit slots; overrides slot_grade
    freeze_interface: bool = False  # freeze embedding + forecast head at init
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
        dt_min=cfg.dt_min,
        dt_max=cfg.dt_max,
        lambda_init=cfg.lambda_init,
        geo_proj=cfg.geo_proj,
    )


class Backbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.geo_embed = cfg.geo_embed
        if cfg.geo_embed:
            al = Algebra.get(3, 0)
            if cfg.d_model % al.dim:
                raise ValueError("geo_embed needs d_model divisible by 8")
            self.mv_dim, n_ch = al.dim, cfg.d_model // al.dim
            self.embed = GeoEmbed(
                al, cfg.n_features, n_ch, grade=cfg.slot_grade, blades=cfg.slot_blades
            )
            make_norm = lambda: GradeRMSNorm(al, n_ch)  # noqa: E731
        else:
            self.embed = nn.Linear(cfg.n_features, cfg.d_model)
            make_norm = lambda: nn.RMSNorm(cfg.d_model)  # noqa: E731
        self.norms = nn.ModuleList(make_norm() for _ in range(cfg.n_layers))
        self.blocks = nn.ModuleList(_make_block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = make_norm()

    def _norm(self, norm, h: torch.Tensor) -> torch.Tensor:
        if not self.geo_embed:
            return norm(h)
        b, l, d = h.shape
        return norm(h.reshape(b, l, d // self.mv_dim, self.mv_dim)).reshape(b, l, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, F] -> h: [B, L, d_model]"""
        h = self.embed(x)
        if self.geo_embed:
            h = h.reshape(*h.shape[:-2], -1)
        for norm, block in zip(self.norms, self.blocks):
            h = h + block(self._norm(norm, h))
        return self._norm(self.final_norm, h)


class _GeoHeadFlat(nn.Module):
    """GeoHead on a flat [B, L, d_model] hidden state."""

    def __init__(self, al: Algebra, c_in: int, n_features: int, grade: int,
                 blades: list[int] | None = None):
        super().__init__()
        self.mv_dim = al.dim
        self.head = GeoHead(al, c_in, n_features, grade=grade, blades=blades)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, l, d = h.shape
        return self.head(h.reshape(b, l, d // self.mv_dim, self.mv_dim))


class GeoMamba(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg)
        self.heads = nn.ModuleDict()
        if "forecast" in cfg.heads:
            if cfg.geo_embed:
                al = Algebra.get(3, 0)
                self.heads["forecast"] = _GeoHeadFlat(
                    al, cfg.d_model // al.dim, cfg.n_features, cfg.slot_grade,
                    blades=cfg.slot_blades,
                )
            else:
                self.heads["forecast"] = nn.Linear(cfg.d_model, cfg.n_features)
        if "anomaly" in cfg.heads:
            self.heads["anomaly"] = nn.Linear(cfg.d_model, 1)
        if cfg.freeze_interface:
            # `frozen` control: the input/output interface stays at its random
            # init, so nothing about the representation is learned. Isolates
            # "fewer things to learn" from "the right geometric type".
            for p in self.backbone.embed.parameters():
                p.requires_grad_(False)
            if "forecast" in self.heads:
                for p in self.heads["forecast"].parameters():
                    p.requires_grad_(False)

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
