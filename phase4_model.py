"""phase4_model.py — Phase 4 variant wiring (ZDA_phase4_spec.md v1.0 SS3).

Variants:
  D0p — dense causal MHA, POSITIONAL-MATCHED RoPE: every rotary pair in
        head h turns at the single per-head frequency w_h (same across-
        head ladder as S). Primary baseline.
  D0  — dense causal MHA, standard within-head RoPE ladder. Reference
        only (prices the positional restriction).
  D1  — dense, FLOP-matched dims, positional-matched as D0p.
  S   — K3Attention, true T16 (phase4_layers).
  X   — K3Attention, shuffled tensor (draw seed = run seed).
  Q0  — query-independent attention: score u_h . k_j / sqrt(dh), key-only
        (no-interaction floor). Ladder RoPE on k.

All attention modules expose .scores(x) -> (aux, masked_logits) so the
trainer logs SS6 diagnostics through one interface (aux = r2 for K3
variants, None for dense/Q0).

Run this file for the constructor/forward test and a param report at the
smoke dims.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase4_layers import (K3Attention, frozen_omega_ladder,
                           shuffled_structure_tensor)

# Per-head frequency ladder: FROZEN at spec v1.0 (SS2) as
# omega_h = (2*pi/1024)^(h/(H-1)) — single source of truth is
# phase4_layers.frozen_omega_ladder, shared by S/X (R_8 action, inside
# K3Attention) and D0p/D1/Q0 (per-head single-frequency RoPE, below).
# Every ladder-mode attention module exposes it as the length-H buffer
# `omega_h`; phase4_grid.py preflight [P4] verifies value and identity.


class DenseAttention(nn.Module):
    """Standard causal MHA with RoPE; rope_mode 'ladder' (per-head single
    frequency, positional-matched to S) or 'standard' (within-head
    ladder). query_free=True gives Q0 (key-only scores)."""

    def __init__(self, d_model, n_heads, rope_mode="ladder",
                 query_free=False, causal=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh, self.causal = n_heads, d_model // n_heads, causal
        self.query_free = query_free
        assert self.dh % 2 == 0
        if rope_mode == "ladder":
            self.register_buffer("omega_h", torch.tensor(
                frozen_omega_ladder(n_heads), dtype=torch.float32))
            freqs = self.omega_h.view(n_heads, 1) \
                        .expand(n_heads, self.dh // 2).clone()
        elif rope_mode == "standard":
            f = 10000.0 ** (-torch.arange(self.dh // 2)
                            / (self.dh // 2))
            freqs = f.view(1, -1).expand(n_heads, self.dh // 2).clone()
        else:
            raise ValueError(rope_mode)
        self.register_buffer("freqs", freqs)          # (H, dh/2)
        if query_free:
            self.u = nn.Parameter(torch.randn(n_heads, self.dh)
                                  / math.sqrt(self.dh))
        else:
            self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def _rope(self, u, pos):
        # u: (B,H,T,dh); rotate pairs (2p, 2p+1) by freqs[h,p]*pos
        th = self.freqs.unsqueeze(1) * pos.view(1, -1, 1)   # (H,T,dh/2)
        c, s = torch.cos(th)[None], torch.sin(th)[None]     # (1,H,T,dh/2)
        u1, u2 = u[..., 0::2], u[..., 1::2]
        out = torch.empty_like(u)
        out[..., 0::2] = u1 * c - u2 * s
        out[..., 1::2] = u1 * s + u2 * c
        return out

    def scores(self, x, pos_offset: float = 0.0):
        B, T, _ = x.shape
        xf = x.float()
        k = self.wk(xf).view(B, T, self.h, self.dh).transpose(1, 2)
        pos = torch.arange(T, dtype=torch.float32,
                           device=x.device) + pos_offset
        k = self._rope(k, pos)
        if self.query_free:
            sal = (k * self.u.view(1, self.h, 1, self.dh)).sum(-1) \
                / math.sqrt(self.dh)                        # (B,H,T_key)
            s = sal.unsqueeze(2).expand(B, self.h, T, T).clone()
        else:
            q = self.wq(xf).view(B, T, self.h, self.dh).transpose(1, 2)
            q = self._rope(q, pos)
            s = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        if self.causal:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool,
                                         device=x.device), 1)
            s = s.masked_fill(mask, float("-inf"))
        return None, s

    def forward(self, x, pos_offset: float = 0.0):
        B, T, _ = x.shape
        _, s = self.scores(x, pos_offset)
        att = torch.softmax(s, dim=-1)
        v = self.wv(x.float()).view(B, T, self.h, self.dh).transpose(1, 2)
        y = (att @ v).transpose(1, 2).reshape(B, T, -1)
        return self.wo(y).to(x.dtype)


def build_attention(variant, d_model, n_heads, seed=0):
    if variant == "D0p":
        return DenseAttention(d_model, n_heads, rope_mode="ladder")
    if variant == "D0":
        return DenseAttention(d_model, n_heads, rope_mode="standard")
    if variant == "D1":
        return DenseAttention(d_model, n_heads, rope_mode="ladder")
    if variant == "S":
        return K3Attention(d_model, n_heads)   # frozen ladder is default
    if variant == "X":
        return K3Attention(d_model, n_heads,
                           tensor=shuffled_structure_tensor(seed))
    if variant == "Q0":
        return DenseAttention(d_model, n_heads, rope_mode="ladder",
                              query_free=True)
    raise ValueError(variant)


class Block(nn.Module):
    def __init__(self, variant, d_model, n_heads, mlp_hidden, seed=0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = build_attention(variant, d_model, n_heads, seed)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp_hidden),
                                 nn.GELU(),
                                 nn.Linear(mlp_hidden, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class Phase4Model(nn.Module):
    def __init__(self, variant, vocab_size, d_model, n_heads, n_layers,
                 mlp_hidden, seed=0):
        super().__init__()
        self.variant = variant
        self.emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [Block(variant, d_model, n_heads, mlp_hidden, seed)
             for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


VARIANTS = ("D0p", "D0", "D1", "S", "X", "Q0")

if __name__ == "__main__":
    torch.manual_seed(0)
    V, d, H, L, M = 65, 64, 2, 2, 256
    x = torch.randint(0, V, (2, 32))
    print(f"{'variant':<8}{'params':>10}   forward/loss")
    for v in VARIANTS:
        m = Phase4Model(v, V, d, H, L, M, seed=1337)
        logits = m(x)
        loss = F.cross_entropy(logits.view(-1, V),
                               torch.randint(0, V, (2 * 32,)))
        loss.backward()
        assert torch.isfinite(loss)
        print(f"{v:<8}{m.n_params():>10,}   OK  loss {loss.item():.3f}")

    # Q0 must have query-independent SCORES (probabilities still vary by
    # row through causal renormalization); D0p must not. Compare raw
    # logits over shared support (rows 5.., keys :6).
    mq = Phase4Model("Q0", V, d, H, L, M)
    _, s = mq.blocks[0].attn.scores(mq.emb(x))
    sp = (s[..., 5:, :6] - s[..., 5:6, :6]).abs().max().item()
    print(f"Q0 score row spread (shared support): {sp:.1e} (expect ~0)")
    md = Phase4Model("D0p", V, d, H, L, M)
    _, s = md.blocks[0].attn.scores(md.emb(x))
    sp2 = (s[..., 5:, :6] - s[..., 5:6, :6]).abs().max().item()
    print(f"D0p score row spread (shared support): {sp2:.3f} (expect > 0)")
    assert sp < 1e-6 and sp2 > 1e-3
    print("\nPhase 4 model wiring OK.")
