"""zda_grid.py — self-contained 18-run ZDA grid for Colab/Kaggle GPU.

Phase 3 of the ZDA experiment (ZDA_experiment_spec.md sections 3-5):
6 variants (B0, B1, V1, V2, V3, V4) x 3 seeds on TinyStories, 4k-vocab
BPE, 300M tokens per run. This file is the whole experiment: it bundles
the Phase-1-validated layers (verbatim from zda_layers.py), the model,
the data pipeline, the trainer, and the pre-registered analysis readout.

Quick start (Colab):
    from google.colab import drive; drive.mount('/content/drive')
    !pip -q install datasets tokenizers
    !python zda_grid.py --out_root /content/drive/MyDrive/zda_runs_ts \
                        --data_dir /content/zda_data

Kaggle: enable Internet + GPU in notebook settings, then
    !python zda_grid.py --out_root /kaggle/working/zda_runs_ts

Behavior:
  * Prints the exact param/FLOP match table (spec section 4 requires
    publishing it) and asserts the targets: params within +-1% of B0 for
    V1/V2/V3/V4, FLOPs within +-5% of V1 for B1.
  * Runs preflight invariant checks (ZD annihilation, step-0 V2==B0,
    causality) before training. --skip_checks to disable.
  * Runs variant x seed sequentially. A run with summary.json is SKIPPED;
    a run with ckpt_last.pt RESUMES mid-run (model, optimizer, data-order
    generator, and RNG states are all restored), so Colab preemption
    costs at most one eval interval (~25M tokens). Just re-run the same
    command after a disconnect.
  * Per run: train_log.csv, eval_log.jsonl (val loss + per-layer/head
    beta + beta grad norm every ~25M tokens), ckpt.pt, summary.json
    (incl. length-generalization eval at ctx 256/512/1024, spec sec. 5).
  * When all requested runs are complete, prints the grand table
    (mean +- std over seeds) and the pre-registered spec-section-1
    decision-rule readout (V2 vs B1 at 2x pooled std; V2 vs V3; V1 vs V4).
    NOTE: after a resume, an eval line can appear twice in eval_log.jsonl
    (crash between eval-log and checkpoint write); keep the last line
    per step when analyzing.

Data notes: TinyStories (HF roneneldan/TinyStories) downloads ~1GB; the
4k BPE is trained on a 250k-story sample of the train split (--bpe_docs),
then the corpus is encoded to uint16 memmaps until --max_train_tokens.
One-time cost ~15 min; cached in --data_dir. Keep --data_dir on fast
local disk (/content), not Drive; keep --out_root on Drive so runs
survive session death. Data order is derived only from (seed), so all
variants at a seed see identical batches — requires the same
--batch_size across the whole grid (default 64).

Diagnostics NOT computed here (post-hoc analysis notebook, from ckpt.pt):
|a_i + c_j| activation distributions and attention entropy (spec sec. 5).

Local debug (no GPU, no datasets/tokenizers needed):
    python zda_grid.py --debug                      # tiny synthetic run
    python zda_grid.py --debug --sim_preempt_step 6 # test resume path
    python zda_grid.py --match_table                # dims/params/flops only
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================================
# SECTION 1 — validated algebra + layers, VERBATIM from zda_layers.py
# (Phase 1: all tests T1-T7 passed). Do not edit this section; any math
# change must go through zda_layers.py + test_zda_numpy.py first.
# ======================================================================

N_ALG = 16  # sedenion dimension


def _cd_conj(x: torch.Tensor) -> torch.Tensor:
    out = -x.clone()
    out[0] = x[0]
    return out


def _cd_mult(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    if n == 1:
        return x * y
    h = n // 2
    a, b = x[:h], x[h:]
    c, d = y[:h], y[h:]
    real = _cd_mult(a, c) - _cd_mult(_cd_conj(d), b)
    imag = _cd_mult(d, a) + _cd_mult(b, _cd_conj(c))
    return torch.cat([real, imag])


def structure_tensor(dim: int = N_ALG, dtype=torch.float64) -> torch.Tensor:
    """T[k, i, j] with (x*y)_k = sum_ij T[k,i,j] x_i y_j."""
    T = torch.zeros(dim, dim, dim, dtype=dtype)
    I = torch.eye(dim, dtype=dtype)
    for i in range(dim):
        for j in range(dim):
            T[:, i, j] = _cd_mult(I[i], I[j])
    return T


def left_mult_matrices(T: torch.Tensor) -> torch.Tensor:
    """L[k][m, j] = T[m, k, j]; (e_k * x)_m = sum_j L[k][m,j] x_j.
    Each L_k is a signed permutation matrix; L_0 = I."""
    return T.permute(1, 0, 2).contiguous()


def shuffled_left_mult_matrices(generator: torch.Generator,
                                dim: int = N_ALG) -> torch.Tensor:
    """V4 control: random signed permutation matrices with L_0 = I
    (preserving the unit-element property, randomizing the algebra)."""
    L = torch.zeros(dim, dim, dim, dtype=torch.float64)
    L[0] = torch.eye(dim, dtype=torch.float64)
    for k in range(1, dim):
        perm = torch.randperm(dim, generator=generator)
        signs = torch.randint(0, 2, (dim,), generator=generator) * 2 - 1
        L[k, torch.arange(dim), perm] = signs.double()
    return L


# Verified annihilating pair: (e3 + e12) * (e5 + e10) = 0 in BOTH orders
# under this repo's Baez convention (verified 2026-07-17, worst-case
# collapse-identity error 4e-16). Chosen over the original (e1+e10, e4-e15)
# because it is KSJ "Pattern 2" — the unique Canonical Six pair that is
# bilateral in both the Cayley-Dickson and Clifford frameworks (AIEX-725).
# The collapse identity (aP + bQ)(bP + cQ) = -2 b (a+c) e0 holds for it.
ZD_PAIR = ((3, +1.0, 12), (5, +1.0, 10))


class PHMLinear(nn.Module):
    """W = sum_k L_k (kron) S_k.  Feature layout is COMPONENT-MAJOR:
    flat index (a * blocks + c) = sedenion component a of block c.
    in_features and out_features must be divisible by 16."""

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, L: torch.Tensor | None = None):
        super().__init__()
        assert in_features % N_ALG == 0 and out_features % N_ALG == 0
        self.in_b = in_features // N_ALG
        self.out_b = out_features // N_ALG
        if L is None:
            L = left_mult_matrices(structure_tensor())
        self.register_buffer("L", L.float())                     # (16,16,16)
        self.S = nn.Parameter(torch.empty(N_ALG, self.out_b, self.in_b))
        # Kaiming-style init scaled for the kron sum:
        # each output unit sums over 16 * in_b inputs with |L| entries = 1.
        nn.init.normal_(self.S, std=1.0 / math.sqrt(N_ALG * self.in_b))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def weight(self) -> torch.Tensor:
        # W[(a,c),(b,d)] = sum_k L[k,a,b] S[k,c,d]
        W = torch.einsum("kab,kcd->acbd", self.L, self.S)
        return W.reshape(N_ALG * self.out_b, N_ALG * self.in_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight(), self.bias)

    def extra_repr(self) -> str:
        return (f"in={N_ALG * self.in_b}, out={N_ALG * self.out_b}, "
                f"params={self.S.numel()} (dense equiv /16)")


def make_proj(kind: str, d_in: int, d_out: int, bias: bool,
              gen: torch.Generator | None = None) -> nn.Module:
    if kind == "dense":
        return nn.Linear(d_in, d_out, bias=bias)
    if kind == "phm16":
        return PHMLinear(d_in, d_out, bias=bias)
    if kind == "phm16_shuffled":
        assert gen is not None, "pass a torch.Generator for reproducibility"
        return PHMLinear(d_in, d_out, bias=bias,
                         L=shuffled_left_mult_matrices(gen))
    raise ValueError(kind)


def rope_cache(seq_len: int, d_head: int, device, base: float = 10000.0):
    assert d_head % 2 == 0
    half = d_head // 2
    freqs = base ** (-torch.arange(0, half, device=device).float() / half)
    t = torch.arange(seq_len, device=device).float()
    ang = torch.outer(t, freqs)                       # (T, half)
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # x: (B, H, T, D). Pairing convention: (x_even, x_odd) rotated jointly.
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin,
                        x1 * sin + x2 * cos], dim=-1).flatten(-2)


class ZDGatedCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 gate: bool = True, frame: str = "zd",
                 learnable_frame: bool = False,
                 proj: str = "dense", dropout: float = 0.0,
                 seed: int = 0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d_h = d_model // n_heads
        assert self.d_h >= N_ALG, "head dim must hold one sedenion block"
        self.gate = gate
        self.dropout = dropout

        gen = torch.Generator().manual_seed(seed)
        self.q_proj = make_proj(proj, d_model, d_model, False, gen)
        self.k_proj = make_proj(proj, d_model, d_model, False, gen)
        self.v_proj = make_proj(proj, d_model, d_model, False, gen)
        self.o_proj = make_proj(proj, d_model, d_model, False, gen)

        if gate:
            p_vec = torch.zeros(self.d_h)
            q_vec = torch.zeros(self.d_h)
            if frame == "zd":
                (i, s1, j), (k, s2, l) = ZD_PAIR
                p_vec[i], p_vec[j] = 1.0, s1      # P = e3 + e12
                q_vec[k], q_vec[l] = 1.0, s2      # Q = e5 + e10
            elif frame == "random":
                g = torch.Generator().manual_seed(seed + 1)
                p_vec = torch.randn(self.d_h, generator=g)
                q_vec = torch.randn(self.d_h, generator=g)
            else:
                raise ValueError(frame)
            p_vec = p_vec / p_vec.norm()
            q_vec = q_vec / q_vec.norm()
            # one frame per head (broadcast copies; independent if learnable)
            P = p_vec.repeat(n_heads, 1)
            Q = q_vec.repeat(n_heads, 1)
            if learnable_frame:
                self.frame_p = nn.Parameter(P)
                self.frame_q = nn.Parameter(Q)
            else:
                self.register_buffer("frame_p", P)
                self.register_buffer("frame_q", Q)
            # beta init 0: exact baseline at init, live gradient
            self.beta = nn.Parameter(torch.zeros(n_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.d_h).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.h, self.d_h).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.h, self.d_h).transpose(1, 2)

        cos, sin = rope_cache(T, self.d_h, x.device)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        logits = q @ k.transpose(-2, -1) / math.sqrt(self.d_h)  # (B,H,T,T)

        if self.gate:
            # a_i = <q_i, p>, c_j = <k_j, q_frame>; gate_ij = |a_i + c_j|
            a = torch.einsum("bhtd,hd->bht", q, self.frame_p)
            c = torch.einsum("bhtd,hd->bht", k, self.frame_q)
            gate = (a.unsqueeze(-1) + c.unsqueeze(-2)).abs()     # (B,H,T,T)
            logits = logits - self.beta.view(1, -1, 1, 1) * gate

        mask = torch.triu(torch.ones(T, T, dtype=torch.bool,
                                     device=x.device), diagonal=1)
        logits = logits.masked_fill(mask, float("-inf"))
        att = F.softmax(logits, dim=-1)
        if self.dropout > 0:
            att = F.dropout(att, self.dropout, self.training)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


# ======================================================================
# SECTION 2 — model + grid dims (adapted from model.py; MLP width is
# configurable because param matching cannot be hit with d_model alone)
# ======================================================================

VARIANT_ATTN_KWARGS = {
    "B0": dict(gate=False, proj="dense"),
    "B1": dict(gate=False, proj="dense"),
    "V1": dict(gate=False, proj="phm16"),
    "V2": dict(gate=True, frame="zd", proj="dense"),
    "V3": dict(gate=True, frame="random", proj="dense"),
    "V4": dict(gate=False, proj="phm16_shuffled"),
}

# Grid dims (spec section 4: 6L / 6H / d_h 64 nominal / ctx 256 / RoPE).
# V1/V4 keep d_model=384 (so d_h stays exactly 64, as the spec fixes) and
# recover the PHM 16x attention-param saving in the MLP width instead.
# B1 is FLOP-matched to V1/V4 by widening d_model (params intentionally
# larger — that is the "fair fight" ceiling).
GRID_DIMS = {
    "B0": dict(d_model=384, mlp_hidden=1536),
    "B1": dict(d_model=432, mlp_hidden=1728),
    "V1": dict(d_model=384, mlp_hidden=2255),
    "V2": dict(d_model=384, mlp_hidden=1536),
    "V3": dict(d_model=384, mlp_hidden=1536),
    "V4": dict(d_model=384, mlp_hidden=2255),
}


@dataclass
class ZDAConfig:
    variant: str
    vocab_size: int = 4096
    n_layers: int = 6
    n_heads: int = 6
    d_model: int = 384
    mlp_hidden: int | None = None      # None -> 4 * d_model
    ctx: int = 256
    dropout: float = 0.0
    seed: int = 1337

    @property
    def mlp(self) -> int:
        return self.mlp_hidden if self.mlp_hidden else 4 * self.d_model


def grid_config(variant: str, seed: int, vocab_size: int) -> ZDAConfig:
    return ZDAConfig(variant=variant, seed=seed, vocab_size=vocab_size,
                     **GRID_DIMS[variant])


class ZDABlock(nn.Module):
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
        # kron-aware init (it has no nn.Linear children).
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
        return {i: b.attn.beta.detach().cpu().tolist()
                for i, b in enumerate(self.blocks)
                if hasattr(b.attn, "beta")}

    def beta_grad_norms(self) -> dict[int, float]:
        return {i: float(b.attn.beta.grad.norm())
                for i, b in enumerate(self.blocks)
                if hasattr(b.attn, "beta") and b.attn.beta.grad is not None}

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


def match_table(vocab_size: int = 4096, check: bool = True) -> str:
    rows = []
    for v in ("B0", "B1", "V1", "V2", "V3", "V4"):
        cfg = grid_config(v, 1337, vocab_size)
        torch.manual_seed(cfg.seed)
        rows.append((v, cfg.d_model, cfg.d_model // cfg.n_heads, cfg.mlp,
                     ZDAModel(cfg).n_params(), flops_per_token(cfg)))
    p_ref = next(r[4] for r in rows if r[0] == "B0")
    f_ref = next(r[5] for r in rows if r[0] == "V1")
    lines = [f"{'variant':8s} {'d_model':>7s} {'d_h':>4s} {'mlp':>5s} "
             f"{'params':>11s} {'vs B0':>8s} {'flops/tok':>11s} {'vs V1':>8s}",
             "-" * 70]
    for v, d, dh, m, p, fl in rows:
        lines.append(f"{v:8s} {d:7d} {dh:4d} {m:5d} {p:11,d} "
                     f"{100*(p/p_ref-1):+7.2f}% {fl:11,d} "
                     f"{100*(fl/f_ref-1):+7.2f}%")
        if check:
            if v in ("V1", "V2", "V3", "V4"):
                assert abs(p / p_ref - 1) < 0.01, \
                    f"{v} params off B0 by >1%: {p:,} vs {p_ref:,}"
            if v == "B1":
                assert abs(fl / f_ref - 1) < 0.05, \
                    f"B1 FLOPs off V1 by >5%: {fl:,} vs {f_ref:,}"
    return "\n".join(lines)


# ======================================================================
# SECTION 3 — data: TinyStories -> 4k BPE -> uint16 memmap bins
# ======================================================================

EOT_TOKEN = "<|endoftext|>"


def prepare_tinystories(data_dir: str, bpe_docs: int,
                        max_train_tokens: int, max_val_tokens: int) -> dict:
    """One-time: train 4k BPE on a TinyStories sample, encode corpus to
    train.bin/val.bin (uint16). Cached — reruns are instant. Imports of
    datasets/tokenizers are lazy so --debug works without them."""
    os.makedirs(data_dir, exist_ok=True)
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)

    try:
        from datasets import load_dataset
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    except ImportError:
        sys.exit("Missing deps for data prep: pip install datasets tokenizers")

    print("[data] loading TinyStories (HF roneneldan/TinyStories) ...")
    dsets = load_dataset("roneneldan/TinyStories")

    tok_path = os.path.join(data_dir, "tok4096.json")
    if not os.path.exists(tok_path):
        print(f"[data] training 4k byte-level BPE on {bpe_docs:,} stories ...")
        tok = Tokenizer(BPE())
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tok.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(vocab_size=4096, special_tokens=[EOT_TOKEN],
                             initial_alphabet=ByteLevel.alphabet(),
                             show_progress=True)
        n = min(bpe_docs, len(dsets["train"]))
        tok.train_from_iterator(
            (dsets["train"][i]["text"] for i in range(n)),
            trainer=trainer, length=n)
        tok.save(tok_path)
    tok = Tokenizer.from_file(tok_path)
    vocab_size = tok.get_vocab_size()
    eot_id = tok.token_to_id(EOT_TOKEN)
    assert vocab_size <= 65535 and eot_id is not None

    def encode_split(split: str, cap: int, out_name: str) -> int:
        out_path = os.path.join(data_dir, out_name)
        buf = np.empty(cap, dtype=np.uint16)
        n_tok, i, bs = 0, 0, 2000
        data = dsets[split]
        print(f"[data] encoding {split} (cap {cap:,} tokens) ...")
        while i < len(data) and n_tok < cap:
            texts = data[i:i + bs]["text"]
            i += bs
            for enc in tok.encode_batch(texts):
                ids = enc.ids + [eot_id]
                take = min(len(ids), cap - n_tok)
                buf[n_tok:n_tok + take] = np.asarray(ids[:take],
                                                     dtype=np.uint16)
                n_tok += take
                if n_tok >= cap:
                    break
            if (i // bs) % 50 == 0:
                print(f"[data]   {split}: {n_tok:,} tokens ...", flush=True)
        buf[:n_tok].tofile(out_path)
        return n_tok

    n_train = encode_split("train", max_train_tokens, "train.bin")
    n_val = encode_split("validation", max_val_tokens, "val.bin")
    meta = {"vocab_size": vocab_size, "n_train": n_train, "n_val": n_val,
            "tokenizer": tok_path, "eot_id": eot_id}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[data] done: {n_train:,} train / {n_val:,} val tokens, "
          f"vocab {vocab_size}")
    return meta


class TokenDataset:
    """Seeded random contiguous chunks over memmapped token bins.
    All randomness comes from the caller's generator, so identically
    seeded generators yield identical data order (spec section 4)."""

    def __init__(self, data_dir: str, meta: dict):
        self.vocab_size = meta["vocab_size"]
        self.train = np.memmap(os.path.join(data_dir, "train.bin"),
                               dtype=np.uint16, mode="r")
        self.val = np.memmap(os.path.join(data_dir, "val.bin"),
                             dtype=np.uint16, mode="r")

    def get_batch(self, split: str, batch_size: int, ctx: int,
                  generator: torch.Generator):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - ctx - 1, (batch_size,),
                           generator=generator).tolist()
        x = torch.from_numpy(np.stack(
            [data[i:i + ctx].astype(np.int64) for i in ix]))
        y = torch.from_numpy(np.stack(
            [data[i + 1:i + ctx + 1].astype(np.int64) for i in ix]))
        return x, y


class SyntheticDataset(TokenDataset):
    """--debug: random tokens, no downloads. Plumbing test only."""

    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        rng = np.random.default_rng(0)
        self.train = rng.integers(0, vocab_size, 200_000).astype(np.uint16)
        self.val = rng.integers(0, vocab_size, 50_000).astype(np.uint16)


# ======================================================================
# SECTION 4 — trainer with mid-run checkpoint/resume
# ======================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_at(step: int, tr: dict) -> float:
    warmup, total = tr["warmup_steps"], tr["steps"]
    if step < warmup:
        return tr["lr"] * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return tr["lr_min"] + 0.5 * (tr["lr"] - tr["lr_min"]) * (1 + math.cos(math.pi * t))


def make_optimizer(model: ZDAModel, tr: dict) -> torch.optim.AdamW:
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": tr["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=tr["lr"], betas=(0.9, 0.95))


@torch.no_grad()
def estimate_loss(model, ds, tr, cfg, step, device) -> dict:
    """Eval batches seeded by (seed, step) only -> every variant is
    evaluated on the same batches at the same step. Idempotent."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        g = torch.Generator().manual_seed(
            cfg.seed * 100_000 + step * 10 + (0 if split == "train" else 1))
        losses = []
        for _ in range(tr["eval_iters"]):
            x, y = ds.get_batch(split, tr["batch_size"], cfg.ctx, g)
            _, loss = model(x.to(device), y.to(device))
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


@torch.no_grad()
def length_gen_eval(model, ds, cfg, device, batch_size: int = 8,
                    iters: int = 32) -> dict:
    """Spec section 5 secondary metric: val loss/ppl at ctx 256/512/1024
    (train ctx 256; RoPE makes longer contexts meaningful)."""
    model.eval()
    out = {}
    for ectx in (256, 512, 1024):
        g = torch.Generator().manual_seed(cfg.seed * 1000 + ectx)
        losses = []
        for _ in range(iters):
            x, y = ds.get_batch("val", batch_size, ectx, g)
            _, loss = model(x.to(device), y.to(device))
            losses.append(loss.item())
        m = float(np.mean(losses))
        out[f"ctx{ectx}"] = {"loss": round(m, 6), "ppl": round(math.exp(m), 3)}
    model.train()
    return out


def rng_state_dict(data_gen: torch.Generator) -> dict:
    st = {"data_gen": data_gen.get_state(),
          "torch": torch.get_rng_state(),
          "numpy": np.random.get_state(),
          "python": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def rng_state_load(st: dict, data_gen: torch.Generator):
    data_gen.set_state(st["data_gen"])
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if torch.cuda.is_available() and "cuda" in st:
        torch.cuda.set_rng_state_all(st["cuda"])


def save_ckpt_atomic(path: str, obj: dict):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_one(cfg: ZDAConfig, ds, tr: dict, run_dir: str, device: str,
              sim_preempt_step: int | None = None) -> dict:
    os.makedirs(run_dir, exist_ok=True)
    last_path = os.path.join(run_dir, "ckpt_last.pt")

    set_seed(cfg.seed)
    model = ZDAModel(cfg).to(device)
    opt = make_optimizer(model, tr)
    data_gen = torch.Generator().manual_seed(cfg.seed + 777)

    start_step, elapsed0, resumed = 0, 0.0, False
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        rng_state_load(ck["rng"], data_gen)
        start_step, elapsed0, resumed = ck["step"], ck["elapsed_s"], True
        print(f"[resume] {cfg.variant} seed {cfg.seed}: "
              f"continuing at step {start_step}/{tr['steps']}")

    n_params = model.n_params()
    print(f"variant={cfg.variant} seed={cfg.seed} d_model={cfg.d_model} "
          f"mlp={cfg.mlp} params={n_params:,} "
          f"flops/tok={flops_per_token(cfg):,} device={device}")

    mode = "a" if resumed else "w"
    train_csv = open(os.path.join(run_dir, "train_log.csv"), mode, newline="")
    csv_w = csv.writer(train_csv)
    if not resumed:
        csv_w.writerow(["step", "lr", "train_loss"])
    eval_jsonl = open(os.path.join(run_dir, "eval_log.jsonl"), mode)

    t0 = time.time() - elapsed0
    beta_gnorm: dict[int, float] = {}
    model.train()
    ev = None
    for step in range(start_step, tr["steps"] + 1):
        lr = lr_at(step, tr)
        for group in opt.param_groups:
            group["lr"] = lr

        eval_due = step % tr["eval_interval"] == 0 or step == tr["steps"]
        if eval_due:
            skip_eval = resumed and step == start_step and step > 0
            if not skip_eval:      # resume point: eval already logged
                ev = estimate_loss(model, ds, tr, cfg, step, device)
                rec = {"step": step, "lr": lr,
                       "train_loss_est": round(ev["train"], 6),
                       "val_loss": round(ev["val"], 6),
                       "betas": model.betas(),
                       "beta_grad_norms": beta_gnorm,
                       "tokens": step * tr["batch_size"] * cfg.ctx,
                       "elapsed_s": round(time.time() - t0, 1)}
                eval_jsonl.write(json.dumps(rec) + "\n")
                eval_jsonl.flush()
                print(f"step {step:6d} | lr {lr:.2e} | train {ev['train']:.4f}"
                      f" | val {ev['val']:.4f} | {rec['elapsed_s']:.0f}s",
                      flush=True)
                save_ckpt_atomic(last_path, {
                    "step": step, "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "rng": rng_state_dict(data_gen),
                    "elapsed_s": round(time.time() - t0, 1)})

        if step == tr["steps"]:
            break

        x, y = ds.get_batch("train", tr["batch_size"], cfg.ctx, data_gen)
        _, loss = model(x.to(device), y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
        nxt = step + 1
        if nxt % tr["eval_interval"] == 0 or nxt == tr["steps"]:
            beta_gnorm = model.beta_grad_norms()
        opt.step()

        if step % tr["log_interval"] == 0:
            csv_w.writerow([step, f"{lr:.6e}", f"{loss.item():.6f}"])
            train_csv.flush()

        if sim_preempt_step is not None and nxt >= sim_preempt_step:
            train_csv.close(); eval_jsonl.close()
            print(f"[debug] simulated preemption after step {step}")
            sys.exit(3)

    train_csv.close()
    eval_jsonl.close()

    if ev is None:      # resumed exactly at the final step; recover eval
        ev = estimate_loss(model, ds, tr, cfg, tr["steps"], device)

    lg = length_gen_eval(model, ds, cfg, device)
    final = {"variant": cfg.variant, "seed": cfg.seed, "n_params": n_params,
             "flops_per_token": flops_per_token(cfg),
             "final_val_loss": ev["val"], "final_train_loss": ev["train"],
             "betas": model.betas(), "length_gen": lg,
             "wall_time_s": round(time.time() - t0, 1),
             "config": {**tr, "d_model": cfg.d_model, "mlp": cfg.mlp,
                        "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
                        "ctx": cfg.ctx, "vocab_size": cfg.vocab_size}}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(final, f, indent=2)
    torch.save({"model": model.state_dict(), "config": final["config"],
                "variant": cfg.variant},
               os.path.join(run_dir, "ckpt.pt"))
    if os.path.exists(last_path):
        os.remove(last_path)
    print(f"done: {run_dir}  final val {ev['val']:.4f}  "
          f"len-gen ppl 512/1024: {lg['ctx512']['ppl']}/{lg['ctx1024']['ppl']}"
          f"  ({final['wall_time_s']:.0f}s)")
    return final


# ======================================================================
# SECTION 5 — preflight invariant checks (fast, CPU, before the grid)
# ======================================================================

def preflight(vocab_size: int):
    # [P1] the verified ZD pair annihilates under this torch build
    T = structure_tensor()
    (i, s1, j), (k, s2, l) = ZD_PAIR
    P = torch.zeros(16, dtype=torch.float64); P[i], P[j] = 1.0, s1
    Q = torch.zeros(16, dtype=torch.float64); Q[k], Q[l] = 1.0, s2
    prod = torch.einsum("kij,i,j->k", T, P, Q)
    assert prod.abs().max() < 1e-12, "ZD pair failed to annihilate"
    print("[P1] ZD pair annihilates under this torch build       OK")

    # [P2] step-0 V2 == B0 at grid dims (beta=0 invariant, end to end)
    seed = 1337
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, vocab_size, (4, 256), generator=g)
    y = torch.randint(0, vocab_size, (4, 256), generator=g)
    set_seed(seed)
    m_v2 = ZDAModel(grid_config("V2", seed, vocab_size))
    set_seed(seed)
    m_b0 = ZDAModel(grid_config("B0", seed, vocab_size))
    m_v2.eval(); m_b0.eval()
    with torch.no_grad():
        _, l_v2 = m_v2(x, y)
        _, l_b0 = m_b0(x, y)
    diff = abs(l_v2.item() - l_b0.item())
    assert diff < 1e-5, f"step-0 V2 != B0: diff={diff:.3e}"
    print(f"[P2] step-0 V2 loss == B0 loss (diff {diff:.1e})       OK")

    # [P3] causality under the gate at grid dims
    m = ZDGatedCausalSelfAttention(384, 6, gate=True)
    xa = torch.randn(2, 32, 384)
    xb = xa.clone(); xb[:, 16:, :] += 1.0
    d = (m(xa) - m(xb)).abs().amax(dim=(0, 2))
    assert d[:16].max() < 1e-6, "causal mask leak"
    print("[P3] causal mask holds under gate                     OK")


# ======================================================================
# SECTION 6 — grid runner + pre-registered readout
# ======================================================================

def grand_summary(out_root: str, variants: list[str], seeds: list[int]):
    stats: dict[str, list[dict]] = {}
    for path in glob.glob(os.path.join(out_root, "*", "summary.json")):
        with open(path) as f:
            s = json.load(f)
        stats.setdefault(s["variant"], []).append(s)

    print("\n================ GRAND SUMMARY ================")
    print(f"{'variant':8s} {'n':>2s} {'val loss (mean ± std)':>24s} "
          f"{'ppl@512':>9s} {'ppl@1024':>9s}")
    agg = {}
    for v in ("B0", "B1", "V1", "V2", "V3", "V4"):
        runs = stats.get(v, [])
        if not runs:
            continue
        losses = np.array([r["final_val_loss"] for r in runs])
        agg[v] = (losses.mean(), losses.std(ddof=1) if len(runs) > 1 else 0.0)
        p512 = np.mean([r["length_gen"]["ctx512"]["ppl"] for r in runs
                        if "length_gen" in r])
        p1024 = np.mean([r["length_gen"]["ctx1024"]["ppl"] for r in runs
                         if "length_gen" in r])
        print(f"{v:8s} {len(runs):2d} {losses.mean():14.4f} ± "
              f"{agg[v][1]:.4f} {p512:9.2f} {p1024:9.2f}")

    def rule(a: str, b: str, label: str):
        if a not in agg or b not in agg:
            return
        (ma, sa), (mb, sb) = agg[a], agg[b]
        pooled = math.sqrt((sa ** 2 + sb ** 2) / 2)
        diff = ma - mb
        verdict = ("PASS (a beats b beyond 2x pooled std)"
                   if diff < -2 * pooled else "negative")
        print(f"{label}: {a} - {b} = {diff:+.4f}, "
              f"2x pooled std = {2 * pooled:.4f} -> {verdict}")

    print("\nPre-registered decision rules (spec section 1):")
    rule("V2", "B1", "H2  (V2 must beat B1)")
    rule("V2", "V3", "H3g (V2 vs gate control; ~0 => not ZD-specific)")
    rule("V1", "B0", "H1  (V1 vs param-matched baseline)")
    rule("V1", "V4", "H3s (V1 vs structure control; ~0 => not algebra)")
    print("Report negative results with the same care as positive ones.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out_root", default="runs_ts")
    ap.add_argument("--data_dir", default="data_ts")
    ap.add_argument("--variants", default="b0,b1,v1,v2,v3,v4")
    ap.add_argument("--seeds", default="1337,1338,1339")
    ap.add_argument("--batch_size", type=int, default=64,
                    help="same value for the WHOLE grid (data-order identity)")
    ap.add_argument("--budget", type=int, default=300_000_000,
                    help="train tokens per run (spec: 300M)")
    ap.add_argument("--eval_tokens", type=int, default=25_000_000)
    ap.add_argument("--eval_iters", type=int, default=40)
    ap.add_argument("--device", default=None)
    ap.add_argument("--bpe_docs", type=int, default=250_000)
    ap.add_argument("--max_train_tokens", type=int, default=350_000_000)
    ap.add_argument("--max_val_tokens", type=int, default=20_000_000)
    ap.add_argument("--no_tf32", action="store_true")
    ap.add_argument("--skip_checks", action="store_true")
    ap.add_argument("--match_table", action="store_true",
                    help="print param/FLOP table and exit")
    ap.add_argument("--debug", action="store_true",
                    help="tiny synthetic-data run (plumbing test, no GPU)")
    ap.add_argument("--sim_preempt_step", type=int, default=None,
                    help="debug: exit(3) after N steps to test resume")
    args = ap.parse_args()

    if args.match_table:
        print(match_table())
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if device == "cpu" and not args.debug:
        print("WARNING: no GPU detected — the real grid needs one "
              "(~25-40 GPU-hours). Use --debug for a local plumbing test.")

    if args.debug:
        ds = SyntheticDataset()
        args.batch_size = 4
        args.budget = 16 * args.batch_size * 256          # 16 steps
        args.eval_tokens = 4 * args.batch_size * 256      # eval every 4
        args.eval_iters = 2
        args.out_root = args.out_root + "_debug"
        if args.variants == "b0,b1,v1,v2,v3,v4":
            args.variants = "b0,v2"
        if args.seeds == "1337,1338,1339":
            args.seeds = "1337"
    else:
        meta = prepare_tinystories(args.data_dir, args.bpe_docs,
                                   args.max_train_tokens, args.max_val_tokens)
        ds = TokenDataset(args.data_dir, meta)
        assert ds.vocab_size == 4096, f"vocab {ds.vocab_size} != 4096"

    print(match_table(ds.vocab_size))

    if not args.skip_checks:
        preflight(ds.vocab_size)

    tokens_per_step = args.batch_size * 256
    tr = {"steps": math.ceil(args.budget / tokens_per_step),
          "batch_size": args.batch_size,
          "lr": 3e-4, "lr_min": 3e-5, "warmup_steps": 2000,
          "weight_decay": 0.1, "grad_clip": 1.0,
          "eval_interval": max(1, round(args.eval_tokens / tokens_per_step)),
          "eval_iters": args.eval_iters, "log_interval": 50}
    if args.debug:
        tr["log_interval"] = 1
    print(f"\n{tr['steps']:,} steps/run x {tokens_per_step:,} tokens/step = "
          f"{tr['steps'] * tokens_per_step / 1e6:.1f}M tokens; "
          f"eval every {tr['eval_interval']:,} steps")

    variants = [v.strip().upper() for v in args.variants.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    for v in variants:
        assert v in GRID_DIMS, f"unknown variant {v}"

    os.makedirs(args.out_root, exist_ok=True)
    for seed in seeds:
        for v in variants:
            run_dir = os.path.join(args.out_root, f"{v}_seed{seed}")
            if os.path.exists(os.path.join(run_dir, "summary.json")):
                print(f"=== {v} seed {seed}: already complete, skipping ===")
                continue
            print(f"=== {v} seed {seed} ===")
            cfg = grid_config(v, seed, ds.vocab_size)
            train_one(cfg, ds, tr, run_dir, device,
                      sim_preempt_step=args.sim_preempt_step)

    grand_summary(args.out_root, variants, seeds)


if __name__ == "__main__":
    main()
