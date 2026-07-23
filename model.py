"""model.py — ZDAModel: tok emb -> N x block -> LN -> tied head.

Variant selection via a single config dataclass mapping to spec section 3
(B0, B1, V1, V2, V3, V4). Attention comes from the validated
zda_layers.ZDGatedCausalSelfAttention, used unmodified.

Documented deviation from the HANDOFF's "Block (from zda_layers)":
zda_layers.Block hardcodes the MLP at 4*d_model. The param/FLOP matching
targets (params +-1% of B0, FLOPs +-5% for B1) cannot be hit with d_model
alone because PHM requires d_model % 16 == 0, so ZDABlock below is a
line-for-line copy of zda_layers.Block with a configurable MLP width.
The attention module — where all the validated math lives — is untouched.

`python model.py` prints the param/FLOP match table for configs/*.json.
"""

from __future__ import annotations
import glob
import json
import os
from dataclasses import dataclass, replace, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from zda_layers import ZDGatedCausalSelfAttention

# Variant mapping, verbatim from zda_layers.py header (spec section 3)
VARIANT_ATTN_KWARGS = {
    "B0": dict(gate=False, proj="dense"),
    "B1": dict(gate=False, proj="dense"),
    "V1": dict(gate=False, proj="phm16"),
    "V2": dict(gate=True, frame="zd", proj="dense"),
    "V3": dict(gate=True, frame="random", proj="dense"),
    "V4": dict(gate=False, proj="phm16_shuffled"),
}


@dataclass
class ZDAConfig:
    variant: str
    vocab_size: int = 65
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    mlp_hidden: int | None = None      # None -> 4 * d_model
    ctx: int = 128
    dropout: float = 0.0
    seed: int = 1337

    @property
    def mlp(self) -> int:
        return self.mlp_hidden if self.mlp_hidden else 4 * self.d_model


class ZDABlock(nn.Module):
    """zda_layers.Block with configurable MLP width (see module docstring)."""

    def __init__(self, d_model: int, n_heads: int, mlp_hidden: int,
                 **attn_kwargs):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = ZDGatedCausalSelfAttention(d_model, n_heads, **attn_kwargs)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp_hidden), nn.GELU(),
                                 nn.Linear(mlp_hidden, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ZDAModel(nn.Module):
    def __init__(self, cfg: ZDAConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([
            ZDABlock(cfg.d_model, cfg.n_heads, cfg.mlp,
                     dropout=cfg.dropout, seed=cfg.seed + i,
                     **VARIANT_ATTN_KWARGS[cfg.variant])
            for i in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight     # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        # nanoGPT-style init for dense layers; PHMLinear keeps its own
        # kron-aware init from zda_layers (it has no nn.Linear children).
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    def betas(self) -> dict[int, list[float]]:
        """Per-layer list of per-head beta values (gated variants only)."""
        return {i: b.attn.beta.detach().tolist()
                for i, b in enumerate(self.blocks)
                if hasattr(b.attn, "beta")}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())   # dedups tied head


def flops_per_token(cfg: ZDAConfig) -> int:
    """Forward matmul FLOPs per token (2 * MACs), T = train ctx.
    PHM projections materialize W and run a dense matmul, so their
    forward cost is counted as dense."""
    d, h, T, m = cfg.d_model, cfg.n_heads, cfg.ctx, cfg.mlp
    per_block = 4 * 2 * d * d          # Q, K, V, O projections
    per_block += 2 * 2 * T * d         # QK^T scores + att @ V
    if VARIANT_ATTN_KWARGS[cfg.variant]["gate"]:
        per_block += 2 * 2 * d + 3 * T * h   # a,c dots + outer add/abs/scale
    per_block += 2 * 2 * d * m         # MLP
    return cfg.n_layers * per_block + 2 * d * cfg.vocab_size   # + tied head


def load_config(path: str) -> tuple[ZDAConfig, dict]:
    """Read a configs/*.json file; model fields go into ZDAConfig,
    everything else (training hyperparams) is returned as a dict."""
    with open(path, encoding="utf-8-sig") as f:   # -sig: tolerate BOM
        raw = json.load(f)
    fields = ZDAConfig.__dataclass_fields__
    model_kw = {k: v for k, v in raw.items() if k in fields}
    train_kw = {k: v for k, v in raw.items() if k not in fields}
    return ZDAConfig(**model_kw), train_kw


def match_table(config_dir: str = "configs") -> str:
    rows = []
    for path in sorted(glob.glob(os.path.join(config_dir, "*.json"))):
        cfg, _ = load_config(path)
        torch.manual_seed(cfg.seed)
        model = ZDAModel(cfg)
        rows.append((cfg.variant, cfg.d_model, cfg.mlp,
                     model.n_params(), flops_per_token(cfg)))
    rows.sort(key=lambda r: r[0])
    p_ref = next(r[3] for r in rows if r[0] == "B0")
    f_ref = next(r[4] for r in rows if r[0] == "V1")
    lines = [f"{'variant':8s} {'d_model':>7s} {'mlp':>5s} {'params':>9s} "
             f"{'vs B0':>8s} {'flops/tok':>10s} {'vs V1':>8s}",
             "-" * 62]
    for v, d, m, p, fl in rows:
        lines.append(f"{v:8s} {d:7d} {m:5d} {p:9,d} {100*(p/p_ref-1):+7.2f}% "
                     f"{fl:10,d} {100*(fl/f_ref-1):+7.2f}%")
    return "\n".join(lines)


if __name__ == "__main__":
    print(match_table())
