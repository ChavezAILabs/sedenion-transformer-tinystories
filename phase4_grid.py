"""phase4_grid.py — Phase 4 anchor-task grid: 6 variants x 3 seeds on
TinyStories (ZDA_phase4_spec.md v1.0, gate 6 -> the grid).

Variants: D0p, D0, D1, S, X, Q0 (spec SS3). Anchor task only — H4d is
pre-registered but not executed (PHASE4_gate5_outcome.md); no dial runs
here. Instrument arms (K4 readout, entmax-1.5 probe) are NOT part of
this grid: they run post-grid, compute permitting, via their own
invocations (spec SS3 instrument-arm paragraphs).

Quick start (Colab):
    from google.colab import drive; drive.mount('/content/drive')
    !pip -q install datasets tokenizers
    # upload FOUR files — sedenion_kernel.py (phase4_layers imports it
    # at module level), phase4_layers.py, phase4_model.py, this file:
    !python phase4_grid.py --out_root /content/drive/MyDrive/p4_runs_ts \
                           --data_dir /content/zda_data

Local debug (no GPU, no datasets/tokenizers needed):
    python phase4_grid.py --debug                      # tiny synthetic run
    python phase4_grid.py --debug --sim_preempt_step 6 # test resume path
    python phase4_grid.py --match_table                # dims/params only

Behavior (zda_grid.py pattern, carried forward):
  * Prints the param/FLOP match table and asserts the spec SS3 targets
    (params +-1% of D0p; FLOPs +-5% D1-vs-S).
  * Preflight invariant checks before training (--skip_checks disables):
    ZD-pair annihilation, gamma init/no-decay, K1 guard alive at init,
    end-to-end causality, frozen omega ladder + D0p/S ladder identity,
    long-context forward (length-gen viability).
  * Runs variant x seed sequentially. summary.json => run SKIPPED;
    ckpt_last.pt => run RESUMES (model/opt/data-gen/RNG restored). Just
    re-run the same command after a Colab disconnect. NOTE: after a
    resume an eval line can appear twice in eval_log.jsonl; analysis
    keeps the LAST line per step (grand_summary does this).
  * Per run: train_log.csv, eval_log.jsonl (val loss + full spec-SS6
    diagnostics every eval — r2 min/p5/median pooled AND per-head-on-
    causal-support, frac r2 < 1e-2/1e-4, entropy, gamma, K1-guard row
    spread), ckpt.pt, summary.json (incl. length-gen at ctx 256/512/1024
    and the gate-style diverged/k1_guard_min/wall_s fields).
  * When all runs are complete, prints the grand table and the frozen
    spec-SS5 readout: H4a (S vs D1 val; OR S vs D0p ppl@1024 within-val
    clause), H4b' (S - X, conditional, in the winning metric), H4c
    (per-head p5(r2) step-0 vs final, same head crossing in all seeds).
    D0 and Q0 are printed as reference/floor, never graded.

Deviations from the zda_grid.py pattern, deliberate and documented:
  1. IMPORTS phase4_layers/phase4_model instead of bundling them.
     zda_grid.py's Section 1 is a verbatim copy of zda_layers.py; the
     verbatim-copy rule cannot be satisfied by re-typing code, and
     importing removes copy-drift risk entirely. Colab cost: upload two
     extra files. If the owner wants a single-file bundle, paste the two
     modules verbatim above SECTION 0 in an in-repo session and delete
     the import — do not retype them.
  2. TF32 is DISABLED on CUDA, unconditionally — the exact inverse of
     zda_grid.py's default. Spec SS9.2 hygiene rule; no CLI escape hatch
     is provided on purpose.
  3. No step-0 baseline-equivalence preflight. K3 is mandatory, there is
     no beta=0-style anchor (CLAUDE.md Phase 4 invariants); replacement
     anchors are checked instead.

IN-REPO INTEGRATION CHECKLIST — resolve each before the real grid
(each is also enforced by a loud runtime failure, nothing fails silent):
  [1] DONE 2026-07-22 — GRID_DIMS tuned against real Phase4Model param
      counts (D0p @ mlp 1536 = Phase 3 B0 lineage, 13,784,064 params;
      worst non-D1 deviation +-0.013%); DIMS_TUNED = True; verified
      with `--match_table`.
  [2] DONE 2026-07-22 — flops_per_token() audited op-by-op against the
      actual phase4_layers/phase4_model ops (see its docstring; the
      estimate's placeholder projections and missing H factor on the
      score-path terms were corrected); FLOPS_AUDITED = True, D1-vs-S
      +-5% assert armed (D1 mlp 1992 matches S FLOPs exactly).
  [3] DONE 2026-07-22 — frozen ladder wired: phase4_layers.
      frozen_omega_ladder() is the single source of truth, exposed as
      the length-H buffer `omega_h` on K3Attention (S/X) and ladder-
      mode DenseAttention (D0p/D1/Q0); preflight [P4] finds and
      verifies it (S == D0p identity confirmed at debug dims and H=6).
  [4] DONE 2026-07-22 — preflight [P5] passes: forward at T=1024 clean
      at debug dims (no ctx-bound positional caches; both rotations
      compute from `pos` directly).
  [5] DONE 2026-07-22 — first real execution of the trainer/data
      paths, CPU: `--debug` clean end-to-end (preflight, S + D0p at
      debug dims, evals, len-gen ppl@512/1024, summaries, grand
      summary with the frozen readout correctly deferred on a partial
      grid); `--sim_preempt_step 6` exited 3 after 6 steps as designed,
      and the re-run resumed from the last eval-boundary checkpoint
      with every subsequent eval line BIT-IDENTICAL to the
      uninterrupted run (final val, k1 spread, len-gen ppl all exact) —
      resume is trajectory-faithful, matching the zda_grid.py
      precedent. Checkpoint cadence = eval interval (a preempt loses at
      most eval_interval steps).

Data note: --data_dir defaults to data_ts, the SAME cache zda_grid.py
built for Phase 3 (same TinyStories corpus, same 4k BPE, spec SS4
"unchanged config lineage") — if that cache exists locally or on Drive,
data prep is instant. Keep --data_dir on fast local disk, --out_root on
Drive. Data order derives only from (seed): all variants at a seed see
identical batches; same --batch_size across the whole grid (default 64).
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
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from phase4_model import Phase4Model, VARIANTS

# ======================================================================
# SECTION 0 — canonical variants + frozen constants (spec v1.0)
# ======================================================================

# Canonical variant names come from phase4_model.VARIANTS (the validated
# module owns the spelling); CLI input is matched case-insensitively.
_CANON = {v.lower(): v for v in VARIANTS}
GRID_ORDER = [v for v in ("D0p", "D0", "D1", "S", "X", "Q0") if v in VARIANTS]
assert len(GRID_ORDER) == 6, (
    f"expected the 6 spec-SS3 variants in phase4_model.VARIANTS, got "
    f"{VARIANTS} — grid and model disagree; fix before running")

L_MAX = 1024          # frozen: length-gen eval length anchoring the ladder
LG_CTXS = (256, 512, 1024)   # spec SS5 H4a clause reads ppl@1024
MARGIN_K = 2.0        # standing margin: 2x pooled std (Phase 3 convention)
EXPECTED_N_SEEDS = 3  # spec SS5: "all rules graded on pooled mean/std over
                       # 3 seeds"; H4c bug postmortem 2026-07-25 (see
                       # h4c_readout) -- this is the frozen registered
                       # count, never derived from a CLI --seeds value,
                       # which may legitimately be narrower for a single
                       # training invocation without narrowing what a
                       # valid GRADED readout requires.

DIMS_TUNED = True     # checklist [1] DONE 2026-07-22 — match_table passes
FLOPS_AUDITED = True   # checklist [2] DONE 2026-07-22 — audited op-by-op


def omega_ladder(n_heads: int, l_max: int = L_MAX) -> torch.Tensor:
    """Frozen spec-SS2 v1.0 ladder: geometric, omega_0 = 1 down to
    omega_{H-1} = 2*pi/l_max; identical for S/X (R_8) and D0p/D1
    (per-head single-frequency RoPE)."""
    assert n_heads >= 2
    h = torch.arange(n_heads, dtype=torch.float64)
    return (2 * math.pi / l_max) ** (h / (n_heads - 1))


# ======================================================================
# SECTION 1 — minimal algebra for preflight [P1] only, VERBATIM from
# zda_grid.py Section 1 (itself verbatim from validated zda_layers.py).
# Do not edit; any math change goes through zda_layers.py + mirrors first.
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


# Verified annihilating pair: (e3 + e12) * (e5 + e10) = 0 in BOTH orders
# under this repo's Baez convention (see zda_grid.py Section 1 for the
# full provenance note; RESULTS_phase2_smoke.md SS6 logs the adoption).
ZD_PAIR = ((3, +1.0, 12), (5, +1.0, 10))


# ======================================================================
# SECTION 2 — grid config, dims, match table
# ======================================================================

# Grid dims: Phase 3 lineage (6L / 6H / d_model 384 / ctx 256), spec SS4
# "unchanged config lineage from Phase 3". mlp_hidden is the per-variant
# param-matching knob (Phase 3 precedent); None = NOT YET TUNED —
# checklist [1]. D0p is the param reference (+-1%); D1 is FLOP-matched
# to S (+-5%) and may deviate in params (the "fair fight" analogue of
# Phase 3's B1). n_heads is identical across variants — the frozen
# ladder spans heads, so head count must not vary within the grid.
# Tuned 2026-07-22 against real Phase4Model counts at vocab 4096
# (D0p @ mlp 1536 = Phase 3 B0 lineage reference, 13,784,064 params):
# D0 identical arch (+0.000%); S/X +0.013%; Q0 -0.008%; D1 FLOP-matched
# to S exactly (+0.000% with the audited formula; params +15.3%, free
# by design — the "fair fight" B1 analogue).
GRID_DIMS: dict[str, dict] = {
    "D0p": dict(d_model=384, mlp_hidden=1536),
    "D0":  dict(d_model=384, mlp_hidden=1536),
    "D1":  dict(d_model=384, mlp_hidden=1992),
    "S":   dict(d_model=384, mlp_hidden=1824),
    "X":   dict(d_model=384, mlp_hidden=1824),
    "Q0":  dict(d_model=384, mlp_hidden=1727),
}
GRID_N_LAYERS = 6
GRID_N_HEADS = 6
GRID_CTX = 256

DEBUG_DIMS = dict(d_model=64, mlp_hidden=256)   # gate-4 smoke lineage
DEBUG_N_LAYERS, DEBUG_N_HEADS, DEBUG_CTX = 2, 2, 64


@dataclass
class P4Config:
    variant: str
    seed: int
    vocab_size: int
    d_model: int
    mlp_hidden: int
    n_layers: int = GRID_N_LAYERS
    n_heads: int = GRID_N_HEADS
    ctx: int = GRID_CTX


def grid_config(variant: str, seed: int, vocab_size: int,
                debug: bool = False) -> P4Config:
    if debug:
        return P4Config(variant, seed, vocab_size,
                        n_layers=DEBUG_N_LAYERS, n_heads=DEBUG_N_HEADS,
                        ctx=DEBUG_CTX, **DEBUG_DIMS)
    dims = GRID_DIMS[variant]
    if dims["mlp_hidden"] is None or not DIMS_TUNED:
        sys.exit(f"GRID_DIMS[{variant!r}] not tuned (checklist [1]): fill "
                 f"mlp_hidden for all variants against real param counts, "
                 f"verify with --match_table, then set DIMS_TUNED = True.")
    return P4Config(variant, seed, vocab_size, **dims)


def build_model(cfg: P4Config) -> Phase4Model:
    set_seed(cfg.seed)
    return Phase4Model(cfg.variant, cfg.vocab_size, cfg.d_model,
                       cfg.n_heads, cfg.n_layers, cfg.mlp_hidden,
                       seed=cfg.seed)


def flops_per_token(cfg: P4Config) -> int:
    """Forward matmul FLOPs per token (2 * MACs), T = train ctx.
    AUDITED 2026-07-22 (checklist [2]) op-by-op against the actual
    modules; the original estimate's placeholder projections and missing
    H factor on the score-path terms are corrected:
      dense (D0p/D0/D1)  wq/wk/wv/wo are d->d (4 * 2d^2); QK^T and
                         att@V are T*(d/H) MACs per token per head each
                         (2 * 2*T*d). [phase4_model.DenseAttention]
      Q0                 no wq — wk/wv/wo (3 * 2d^2) + per-head salience
                         u.k (2d) + att@V (2*T*d).
      S/X                wq/wk are d -> H*16 (2 * 2*d*16H); wv/wo d->d
                         (2 * 2d^2); R_8 rotation of q and k is a real
                         16x16 matmul per token per head (2 * H*2*16^2);
                         two-step contraction per K3Attention.scores():
                         k_rot einsum('mij,bhsj->bhsim') = 16^3 MACs per
                         key token per head (H * 2*16^3), pair product
                         einsum('bhti,bhsim->bhtsm') = 16^2 MACs per
                         (t,s) pair per head (H*T * 2*16^2); norms +
                         ratio + gamma ~ (2*16+4) per pair (H*T*36);
                         att@V (2*T*d). [phase4_layers.K3Attention]
      MLP 4*d*m; final head 2*d*V; emb lookup 0.
    Excluded (each sub-1% and common-mode across variants): LayerNorms,
    biases, GELU, softmax, elementwise RoPE rotation, residual adds."""
    d, h, T = cfg.d_model, cfg.n_heads, cfg.ctx
    m = cfg.mlp_hidden
    A = 16
    if cfg.variant in ("S", "X"):
        att = (2 * 2 * d * (h * A)          # wq, wk: d -> H*16
               + 2 * 2 * d * d              # wv, wo
               + 2 * h * 2 * A * A          # R_8 rotate q and k
               + h * 2 * A ** 3             # k_rot build, per key token
               + h * T * 2 * A * A          # pair product, all T keys
               + h * T * (2 * A + 4)        # norms + ratio + gamma
               + 2 * T * d)                 # att @ V
    elif cfg.variant == "Q0":
        att = (3 * 2 * d * d                # wk, wv, wo (no wq)
               + 2 * d                      # salience u . k, all heads
               + 2 * T * d)                 # att @ V
    else:                                   # D0p, D0, D1: dense MHA
        att = (4 * 2 * d * d                # wq, wk, wv, wo
               + 2 * 2 * T * d)             # QK^T + att @ V
    per_block = att + 4 * d * m             # MLP
    return cfg.n_layers * per_block + 2 * d * cfg.vocab_size


def match_table(vocab_size: int = 4096, check: bool = True,
                debug: bool = False) -> str:
    rows = []
    for v in GRID_ORDER:
        cfg = grid_config(v, 1337, vocab_size, debug=debug)
        rows.append((v, cfg.d_model, cfg.mlp_hidden,
                     build_model(cfg).n_params(), flops_per_token(cfg)))
    p_ref = next(r[3] for r in rows if r[0] == "D0p")
    f_ref = next(r[4] for r in rows if r[0] == "S")
    flag = "" if FLOPS_AUDITED else " (UNAUDITED — advisory only)"
    lines = [f"{'variant':8s} {'d_model':>7s} {'mlp':>5s} {'params':>11s} "
             f"{'vs D0p':>8s} {'flops/tok':>11s} {'vs S':>8s}{flag}",
             "-" * 72]
    for v, d, m, p, fl in rows:
        lines.append(f"{v:8s} {d:7d} {m:5d} {p:11,d} "
                     f"{100 * (p / p_ref - 1):+7.2f}% {fl:11,d} "
                     f"{100 * (fl / f_ref - 1):+7.2f}%")
        if check and not debug:
            if v != "D1":       # D1 is the FLOP-matched arm, params free
                assert abs(p / p_ref - 1) < 0.01, \
                    f"{v} params off D0p by >1%: {p:,} vs {p_ref:,}"
            elif FLOPS_AUDITED:
                assert abs(fl / f_ref - 1) < 0.05, \
                    f"D1 FLOPs off S by >5%: {fl:,} vs {f_ref:,}"
    return "\n".join(lines)


# ======================================================================
# SECTION 3 — data: TinyStories -> 4k BPE -> uint16 memmap bins
# (VERBATIM from zda_grid.py Section 3; shares its data_ts cache)
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
    seeded generators yield identical data order."""

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
# SECTION 4 — spec SS6 diagnostics (adapted from phase4_train.diagnose;
# pooled-full stats kept for continuity with the gate-4 smoke logs,
# per-head-on-causal-support p5 ADDED as the graded H4c input)
# ======================================================================

def diagnose(model, x, n_heads: int) -> list[dict]:
    """Walks blocks manually so each attention's scores() sees its true
    input. Same conventions as phase4_train.py for the shared fields."""
    out = []
    with torch.no_grad():
        h = model.emb(x)
        for blk in model.blocks:
            xin = blk.ln1(h)
            aux, s = blk.attn.scores(xin)
            att = torch.softmax(s, dim=-1)
            ent = -(att.clamp_min(1e-12).log() * att).sum(-1)  # (B,H,T)
            d = {"entropy": ent.mean(dim=(0, 2)).tolist()}
            T = s.shape[-1]
            half = T // 2
            spread = (s[..., half:, :half]
                      - s[..., half:half + 1, :half]).abs()
            d["k1_row_spread"] = float(spread.max())
            if aux is not None:                       # K3 variants
                r2 = aux.flatten().float()
                # sort-based nearest-rank quantiles — GPU hotfixes
                # 2026-07-22 (PHASE4_colab_launch_log SS3): the original
                # torch.quantile call built its q tensor on CPU against
                # CUDA input, and torch.quantile has a hard 2^24-element
                # cap < the pooled r2 tensor at grid dims (~25.2M).
                # Pooled fields are DESCRIPTIVE-only; vs torch.quantile's
                # linear interpolation, p5/med shift by <=1 order
                # statistic and r2_min is exact either way.
                n = r2.numel()
                r2s, _ = r2.sort()
                q = torch.stack([r2s[0],
                                 r2s[max(0, int(0.05 * (n - 1)))],
                                 r2s[int(0.5 * (n - 1))]])
                d.update(r2_min=float(q[0]), r2_p5=float(q[1]),
                         r2_med=float(q[2]),
                         frac_r2_lt_1e2=float((r2 < 1e-2).float().mean()),
                         frac_r2_lt_1e4=float((r2 < 1e-4).float().mean()),
                         gamma=blk.attn.gamma.tolist())
                # graded H4c input: per-head p5(r2) on causal support
                # (realized pairs only — upper triangle is never attended)
                assert aux.dim() == 4 and aux.shape[1] == n_heads, (
                    f"expected aux (B,H,T,T), got {tuple(aux.shape)} — "
                    f"per-head H4c stats need the head axis; adapt here "
                    f"if phase4_layers changes the aux contract")
                tril = torch.tril(torch.ones(T, T, dtype=torch.bool,
                                             device=aux.device))
                ah = aux.permute(1, 0, 2, 3)[:, :, tril].reshape(
                    n_heads, -1).float()              # (H, B*n_tril)
                p5h = torch.quantile(ah, 0.05, dim=1)
                medh = torch.quantile(ah, 0.5, dim=1)
                d["r2_p5_head"] = p5h.tolist()
                d["r2_med_head"] = medh.tolist()
            out.append(d)
            h = blk(h)
    return out


# ======================================================================
# SECTION 5 — trainer with mid-run checkpoint/resume
# (zda_grid.py Section 4 pattern; loss computed externally because
# Phase4Model.forward returns logits only, per phase4_train.py)
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
    return tr["lr_min"] + 0.5 * (tr["lr"] - tr["lr_min"]) * (
        1 + math.cos(math.pi * t))


def make_optimizer(model, tr: dict) -> torch.optim.AdamW:
    # >=2-D params decay; gamma (1-D) is never decayed — same rule and
    # same reason as beta in Phase 1-3 (CLAUDE.md Phase 4 invariants).
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": tr["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=tr["lr"], betas=(0.9, 0.95))


def ce_loss(model, x, y) -> torch.Tensor:
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           y.reshape(-1))


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
            losses.append(ce_loss(model, x.to(device), y.to(device)).item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


@torch.no_grad()
def length_gen_eval(model, ds, cfg, device, batch_size: int = 8,
                    iters: int = 32) -> dict:
    """H4a secondary metric: val loss/ppl at ctx 256/512/1024."""
    model.eval()
    out = {}
    for ectx in LG_CTXS:
        g = torch.Generator().manual_seed(cfg.seed * 1000 + ectx)
        losses = []
        for _ in range(iters):
            x, y = ds.get_batch("val", batch_size, ectx, g)
            losses.append(ce_loss(model, x.to(device), y.to(device)).item())
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


def train_one(cfg: P4Config, ds, tr: dict, run_dir: str, device: str,
              sim_preempt_step: int | None = None) -> dict:
    os.makedirs(run_dir, exist_ok=True)
    last_path = os.path.join(run_dir, "ckpt_last.pt")

    model = build_model(cfg).to(device)
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
          f"mlp={cfg.mlp_hidden} params={n_params:,} device={device}")

    mode = "a" if resumed else "w"
    train_csv = open(os.path.join(run_dir, "train_log.csv"), mode, newline="")
    csv_w = csv.writer(train_csv)
    if not resumed:
        csv_w.writerow(["step", "lr", "train_loss"])
    eval_jsonl = open(os.path.join(run_dir, "eval_log.jsonl"), mode)

    t0 = time.time() - elapsed0
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
                g = torch.Generator().manual_seed(cfg.seed * 100_000
                                                 + step * 10 + 1)
                xb, _ = ds.get_batch("val", tr["batch_size"], cfg.ctx, g)
                diag = diagnose(model, xb.to(device), cfg.n_heads)
                k1 = min(d["k1_row_spread"] for d in diag)
                rec = {"step": step, "lr": lr,
                       "train_loss_est": round(ev["train"], 6),
                       "val_loss": round(ev["val"], 6),
                       "k1_row_spread": k1,
                       "tokens": step * tr["batch_size"] * cfg.ctx,
                       "elapsed_s": round(time.time() - t0, 1),
                       "diag": diag}
                eval_jsonl.write(json.dumps(rec) + "\n")
                eval_jsonl.flush()
                print(f"step {step:6d} | lr {lr:.2e} | train {ev['train']:.4f}"
                      f" | val {ev['val']:.4f} | k1 {k1:.2e} | "
                      f"{rec['elapsed_s']:.0f}s", flush=True)
                save_ckpt_atomic(last_path, {
                    "step": step, "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "rng": rng_state_dict(data_gen),
                    "elapsed_s": round(time.time() - t0, 1)})

        if step == tr["steps"]:
            break

        x, y = ds.get_batch("train", tr["batch_size"], cfg.ctx, data_gen)
        loss = ce_loss(model, x.to(device), y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
        opt.step()

        if step % tr["log_interval"] == 0:
            csv_w.writerow([step, f"{lr:.6e}", f"{loss.item():.6f}"])
            train_csv.flush()

        if not torch.isfinite(loss):
            print("DIVERGED: non-finite train loss — stopping this run")
            break

        nxt = step + 1
        if sim_preempt_step is not None and nxt >= sim_preempt_step:
            train_csv.close(); eval_jsonl.close()
            print(f"[debug] simulated preemption after step {step}")
            sys.exit(3)

    train_csv.close()
    eval_jsonl.close()

    if ev is None:      # resumed exactly at the final step; recover eval
        ev = estimate_loss(model, ds, tr, cfg, tr["steps"], device)

    # go/no-go fields from the FULL eval log (robust across resumes):
    # keep last record per step, then val0 / k1_guard_min / diverged
    recs = read_eval_log(os.path.join(run_dir, "eval_log.jsonl"))
    val0 = recs[0]["val_loss"] if recs else ev["val"]
    k1_min = min((r["k1_row_spread"] for r in recs), default=float("nan"))
    diverged = (not np.isfinite(ev["val"])) or ev["val"] > 1.2 * val0

    lg = length_gen_eval(model, ds, cfg, device)
    final = {"variant": cfg.variant, "seed": cfg.seed, "n_params": n_params,
             "final_val_loss": ev["val"], "final_train_loss": ev["train"],
             "val0": val0, "diverged": bool(diverged),
             "k1_guard_min": k1_min, "length_gen": lg,
             "wall_time_s": round(time.time() - t0, 1),
             "config": {**tr, "d_model": cfg.d_model,
                        "mlp_hidden": cfg.mlp_hidden,
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
          f"  diverged={diverged}  k1_min={k1_min:.2e}"
          f"  ({final['wall_time_s']:.0f}s)")
    return final


# ======================================================================
# SECTION 6 — preflight invariant checks (fast, CPU, before the grid).
# Replacement anchors per spec SS7 — deliberately NO beta=0-style
# baseline-equivalence check (closed door for a mandatory kernel).
# ======================================================================

_LADDER_ATTRS = ("omega_h", "omega", "omegas", "freqs", "ladder")


def _find_ladder(attn) -> torch.Tensor | None:
    for name in _LADDER_ATTRS:
        v = getattr(attn, name, None)
        if torch.is_tensor(v) and v.numel() >= 2:
            return v.detach().flatten().double()
    return None


def preflight(vocab_size: int, debug: bool):
    # [P1] the verified ZD pair annihilates under this torch build
    T = structure_tensor()
    (i, s1, j), (k, s2, l) = ZD_PAIR
    P = torch.zeros(16, dtype=torch.float64); P[i], P[j] = 1.0, s1
    Q = torch.zeros(16, dtype=torch.float64); Q[k], Q[l] = 1.0, s2
    prod = torch.einsum("kij,i,j->k", T, P, Q)
    assert prod.abs().max() < 1e-12, "ZD pair failed to annihilate"
    print("[P1] ZD pair annihilates under this torch build       OK")

    cfg_s = grid_config("S", 1337, vocab_size, debug=debug)
    m_s = build_model(cfg_s)
    cfg_d = grid_config("D0p", 1337, vocab_size, debug=debug)
    m_d = build_model(cfg_d)

    # [P2] gamma: init 1.0 (frozen SS2), lives in the no-decay group
    for li, blk in enumerate(m_s.blocks):
        g = blk.attn.gamma.detach()
        assert torch.allclose(g, torch.ones_like(g)), \
            f"layer {li} gamma init != 1.0: {g.tolist()} (frozen SS2 says 1.0)"
        assert blk.attn.gamma.dim() < 2, "gamma must be 1-D (no weight decay)"
    print("[P2] gamma init 1.0, 1-D (never weight-decayed)       OK")

    # [P3] K1 guard alive at init + end-to-end causality at these dims
    m_s.eval()
    gen = torch.Generator().manual_seed(7)
    xa = torch.randint(0, vocab_size, (2, cfg_s.ctx), generator=gen)
    xb = xa.clone()
    xb[:, cfg_s.ctx // 2:] = torch.randint(0, vocab_size,
                                           (2, cfg_s.ctx // 2), generator=gen)
    with torch.no_grad():
        diag = diagnose(m_s, xa, cfg_s.n_heads)
        assert min(d["k1_row_spread"] for d in diag) > 0, \
            "K1 guard dead at init: query-independent score rows"
        la, lb = m_s(xa), m_s(xb)
    d_first = (la - lb)[:, :cfg_s.ctx // 2, :].abs().max()
    assert d_first < 1e-5, f"causal leak: first-half logits moved {d_first:.2e}"
    print("[P3] K1 guard alive at init; causality holds          OK")

    # [P4] frozen omega ladder wired in-layer, identical S vs D0p
    lad_s = _find_ladder(m_s.blocks[0].attn)
    lad_d = _find_ladder(m_d.blocks[0].attn)
    expect = omega_ladder(cfg_s.n_heads)
    if lad_s is None or lad_d is None:
        raise RuntimeError(
            "[P4] cannot locate the per-head frequency ladder on the "
            f"attention module (tried attrs {_LADDER_ATTRS}). Checklist "
            "[3]: wire the frozen spec-SS2 ladder omega_h = "
            "(2*pi/1024)**(h/(H-1)) into phase4_layers for S/X and "
            "D0p/D1 identically, exposed as a length-H tensor attribute "
            "named 'omega_h'. Refusing to run an unverifiable grid; "
            "--skip_checks overrides (NOT for the real grid).")
    assert lad_s.numel() == cfg_s.n_heads, \
        f"S ladder has {lad_s.numel()} entries, expected H={cfg_s.n_heads}"
    assert torch.allclose(lad_s, expect, rtol=1e-6), \
        f"S ladder != frozen SS2 ladder:\n  got {lad_s.tolist()}\n  " \
        f"want {expect.tolist()}"
    assert torch.allclose(lad_s, lad_d, rtol=1e-9), \
        "S and D0p ladders differ — positional-matching invariant broken"
    print("[P4] frozen omega ladder verified, S == D0p           OK")

    # [P5] long-context forward (length-gen at 1024 must be possible)
    xl = torch.randint(0, vocab_size, (1, L_MAX), generator=gen)
    with torch.no_grad():
        out = m_s(xl)
    assert out.shape[1] == L_MAX and torch.isfinite(out).all(), \
        "forward at T=1024 failed/non-finite — checklist [4]"
    print(f"[P5] forward at T={L_MAX} (length-gen viable)          OK")


# ======================================================================
# SECTION 7 — grand summary + frozen spec SS5 readout
# ======================================================================

def read_eval_log(path: str) -> list[dict]:
    """Last record per step (resume can duplicate an eval line)."""
    if not os.path.exists(path):
        return []
    by_step: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_step[r["step"]] = r
    return [by_step[s] for s in sorted(by_step)]


def _pooled_margin(sa: float, sb: float) -> float:
    return MARGIN_K * math.sqrt((sa ** 2 + sb ** 2) / 2)


def h4c_readout(out_root: str, seeds: list[int], variant: str = "S",
                graded: bool = True):
    """Frozen H4c rule: pass iff >=1 (layer, head) whose end-of-training
    p5(r2) is below its step-0 p5(r2) by more than 2x the pooled-across-
    seeds std of that head's step-0 p5, in ALL seeds (same head).
    graded=False (the X trace): same computation, descriptive language —
    X's trace is against its OWN null structure and is never graded.

    BUG FIX 2026-07-25 (postmortem: H4c nan-margin/false-"negative"
    incident, first seen on the completed 18-run grid, phase4_grid.py
    sha256 65d419e05b6ec74d...). Root cause: this function used to loop
    `for seed in seeds` over the *passed-in* seeds argument, which
    grand_summary in turn got from main()'s `--seeds` CLI value — the
    same value that also controls which seeds that invocation's TRAINING
    loop iterates. Those are two different concerns: a later invocation
    legitimately narrowing --seeds to finish one remaining seed (e.g.
    `--seeds 1339`, since 1337/1338 already had summary.json and would
    be skipped regardless) silently propagated into the READOUT too,
    so h4c_readout pooled std(ddof=1) over a single-seed axis
    (N=1, ddof=1 => N-ddof=0 => nan, matching the numpy "degrees of
    freedom <= 0" warning exactly). nan comparisons are always False in
    IEEE754, so `crossed` was False everywhere and "negative" printed —
    a nan fall-through, not a genuine test of any head. Spec SS5 already
    states "No grading from partial seeds"; this was a violation of that
    rule via a code path that didn't enforce it. Compare the H4a/H4b'
    path in grand_summary, which was never affected because it
    auto-discovers every summary.json via glob rather than trusting the
    CLI seeds list — h4c_readout now does the analogous thing: seeds are
    discovered from the run directories actually present on disk for
    THIS variant, and grading refuses (does not crash) unless exactly
    EXPECTED_N_SEEDS are found. `seeds` is kept as a parameter only for
    a diagnostic cross-check against what's discovered; it is never used
    to enumerate what gets read."""
    run_dirs = sorted(glob.glob(os.path.join(out_root, f"{variant}_seed*")))
    discovered = sorted(
        int(os.path.basename(d).rsplit("_seed", 1)[1]) for d in run_dirs)
    if seeds and sorted(seeds) != discovered:
        print(f"H4c ({variant}): NOTE — the seeds argument this function "
              f"received ({sorted(seeds)}) differs from the seeds actually "
              f"present on disk ({discovered}). Grading uses the "
              f"discovered set. (This note existing, and grading not "
              f"silently using the passed-in list, is the 2026-07-25 fix.)")
    if len(discovered) != EXPECTED_N_SEEDS:
        print(f"H4c ({variant}): {len(discovered)} seed(s) present on disk "
              f"{discovered}, need exactly {EXPECTED_N_SEEDS} (spec SS5: "
              f"\"No grading from partial seeds\") — cannot grade")
        return None
    p0s, pTs = [], []
    for seed in discovered:
        recs = read_eval_log(os.path.join(out_root, f"{variant}_seed{seed}",
                                          "eval_log.jsonl"))
        if len(recs) < 2 or "r2_p5_head" not in recs[0]["diag"][0]:
            print(f"H4c ({variant}): incomplete per-head traces "
                  f"(seed {seed}) — cannot grade")
            return None
        p0s.append([d["r2_p5_head"] for d in recs[0]["diag"]])
        pTs.append([d["r2_p5_head"] for d in recs[-1]["diag"]])
    p0 = np.array(p0s)   # (seeds, layers, heads)
    pT = np.array(pTs)
    assert p0.shape[0] == EXPECTED_N_SEEDS, (
        f"internal invariant violated: {p0.shape[0]} rows collected but "
        f"{EXPECTED_N_SEEDS} seeds were discovered — this should be "
        f"impossible given the loop above; investigate before trusting "
        f"anything downstream")
    margin = MARGIN_K * p0.std(axis=0, ddof=1)         # (layers, heads)
    # Robustness fix (generalizable beyond Phase 4, per the postmortem):
    # a verdict must never be emitted from a non-finite margin. Raise
    # rather than let a downstream comparison silently fall through to
    # False (nan is never < or > anything) and print a false "negative".
    if not np.all(np.isfinite(margin)):
        bad = [(int(l), int(h))
               for l, h in zip(*np.nonzero(~np.isfinite(margin)))]
        raise RuntimeError(
            f"H4c ({variant}): non-finite margin at (layer,head) {bad} "
            f"despite {EXPECTED_N_SEEDS} seeds present and validated — "
            f"this should not be possible; do not catch this and grade "
            f"anyway, investigate the underlying p0 values first")
    crossed = pT < (p0 - margin[None])                 # per seed
    all_seeds = crossed.all(axis=0)                    # (layers, heads)
    hits = [(int(l), int(h)) for l, h in zip(*np.nonzero(all_seeds))]
    if graded:
        verdict = "PASS" if hits else "negative"
        print(f"H4c ({variant}, per-head p5(r2) step-0 vs final, margin "
              f"{MARGIN_K:g}x pooled step-0 std, same head in all seeds): "
              f"{verdict}" + (f" — heads {hits}" if hits else ""))
    else:
        print(f"{variant} manifold trace (descriptive, own null "
              f"structure, NOT graded): "
              + (f"heads {hits} moved beyond the H4c-style margin"
                 if hits else "no head moved beyond the H4c-style margin"))

    def line(l, h, mark=""):
        print(f"  L{l}H{h}: p5 {p0[:, l, h].mean():.4f} -> "
              f"{pT[:, l, h].mean():.4f} (margin {margin[l, h]:.4f}){mark}")
    for l, h in hits:
        line(l, h, "  <-- crossed, all seeds")
    drops = (p0.mean(axis=0) - pT.mean(axis=0))        # (layers, heads)
    order = np.dstack(np.unravel_index(np.argsort(drops, axis=None)[::-1],
                                       drops.shape))[0]
    shown = 0
    for l, h in order:                                 # top movers for context
        if (int(l), int(h)) in hits:
            continue
        line(int(l), int(h))
        shown += 1
        if shown >= 5:
            break
    return bool(hits)


def grand_summary(out_root: str, variants: list[str], seeds: list[int]):
    stats: dict[str, list[dict]] = {}
    for path in glob.glob(os.path.join(out_root, "*", "summary.json")):
        with open(path) as f:
            s = json.load(f)
        stats.setdefault(s["variant"], []).append(s)

    print("\n================ GRAND SUMMARY ================")
    print(f"{'variant':8s} {'n':>2s} {'val loss (mean ± std)':>24s} "
          f"{'ppl@512':>9s} {'ppl@1024':>9s} {'k1 min':>9s} {'wall h':>7s}")
    agg, agg_ppl, wall = {}, {}, {}
    for v in GRID_ORDER:
        runs = stats.get(v, [])
        if not runs:
            continue
        losses = np.array([r["final_val_loss"] for r in runs])
        ppl1024 = np.array([r["length_gen"]["ctx1024"]["ppl"] for r in runs
                            if "length_gen" in r])
        agg[v] = (losses.mean(), losses.std(ddof=1) if len(runs) > 1 else 0.0)
        agg_ppl[v] = (ppl1024.mean(),
                      ppl1024.std(ddof=1) if len(ppl1024) > 1 else 0.0)
        wall[v] = np.mean([r["wall_time_s"] for r in runs])
        p512 = np.mean([r["length_gen"]["ctx512"]["ppl"] for r in runs
                        if "length_gen" in r])
        k1m = min(r.get("k1_guard_min", float("nan")) for r in runs)
        div = [r for r in runs if r.get("diverged")]
        print(f"{v:8s} {len(runs):2d} {losses.mean():14.4f} ± "
              f"{agg[v][1]:.4f} {p512:9.2f} {agg_ppl[v][0]:9.2f} "
              f"{k1m:9.2e} {wall[v] / 3600:7.2f}"
              + ("  DIVERGED RUNS PRESENT" if div else ""))

    complete = all(len(stats.get(v, [])) >= len(seeds) for v in GRID_ORDER)
    if not complete:
        print("\n(grid incomplete — frozen readout deferred; no grading "
              "from partial seeds)")
        return

    print("\nFrozen decision rules (spec SS5, v1.0). D0 = reference only; "
          "Q0 = no-interaction floor; neither is graded.")

    # ---- H4a ----
    (mS, sS), (mD1, sD1) = agg["S"], agg["D1"]
    m_val_SD1 = _pooled_margin(sS, sD1)
    a1 = (mS - mD1) < -m_val_SD1
    print(f"H4a-1: S - D1 val = {mS - mD1:+.4f}, margin {m_val_SD1:.4f} "
          f"-> {'PASS' if a1 else 'negative'}")

    (mDp, sDp) = agg["D0p"]
    (pS, psS), (pDp, psDp) = agg_ppl["S"], agg_ppl["D0p"]
    m_ppl = _pooled_margin(psS, psDp)
    m_val_SDp = _pooled_margin(sS, sDp)
    a2 = ((pS - pDp) < -m_ppl) and (abs(mS - mDp) <= m_val_SDp)
    print(f"H4a-2: S - D0p ppl@1024 = {pS - pDp:+.3f} (margin {m_ppl:.3f}) "
          f"with |S - D0p| val = {abs(mS - mDp):.4f} "
          f"(within {m_val_SDp:.4f}) -> {'PASS' if a2 else 'negative'}")
    h4a = a1 or a2
    win_metric = "val" if a1 else ("ppl1024" if a2 else None)
    print(f"H4a: {'POSITIVE' if h4a else 'NEGATIVE'}")

    # ---- H4b' (conditional on an H4a win, in the winning metric) ----
    if h4a:
        if win_metric == "val":
            (mX, sX) = agg["X"]
            diff, marg = mS - mX, _pooled_margin(sS, sX)
            ok = diff < -marg
        else:
            (pX, psX) = agg_ppl["X"]
            diff, marg = pS - pX, _pooled_margin(psS, psX)
            ok = diff < -marg
        print(f"H4b': S - X ({win_metric}) = {diff:+.4f}, margin {marg:.4f} "
              f"-> {'PASS (ZD-structure-specific)' if ok else 'negative '
              '(generic K3-family effect, not ZD geometry)'}")
    else:
        print("H4b': not evaluated (conditional on an H4a win)")

    # ---- H4c ----
    h4c = h4c_readout(out_root, seeds, "S")
    h4c_readout(out_root, seeds, "X", graded=False)

    # ---- bookkeeping the spec asks for ----
    if "S" in wall and "D0p" in wall and wall["D0p"] > 0:
        r = wall["S"] / wall["D0p"]
        print(f"Wall-clock S/D0p = {r:.2f}x "
              f"({'within' if r < 2 else 'EXCEEDS'} the 2x bound)")
    if h4a:
        print("NOTE: H4a positive — the pre-registered H4d resumption "
              "condition is met (spec SS4); the dial becomes worth "
              "resuming investment in.")
    if h4a and h4c is False:
        print("NOTE: S win with H4c negative — report as real-but-"
              "mechanism-unattributed (spec SS5).")
    print("Report negative results with the same care as positive ones.")


# ======================================================================
# SECTION 8 — main
# ======================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out_root", default="runs_p4_ts")
    ap.add_argument("--data_dir", default="data_ts",
                    help="shared with zda_grid.py — reuse the Phase 3 cache")
    ap.add_argument("--variants", default=",".join(GRID_ORDER))
    ap.add_argument("--seeds", default="1337,1338,1339")
    ap.add_argument("--batch_size", type=int, default=64,
                    help="same value for the WHOLE grid (data-order identity)")
    ap.add_argument("--budget", type=int, default=300_000_000,
                    help="train tokens per run (Phase 3 lineage: 300M)")
    ap.add_argument("--eval_tokens", type=int, default=25_000_000)
    ap.add_argument("--eval_iters", type=int, default=40)
    ap.add_argument("--device", default=None)
    ap.add_argument("--bpe_docs", type=int, default=250_000)
    ap.add_argument("--max_train_tokens", type=int, default=350_000_000)
    ap.add_argument("--max_val_tokens", type=int, default=20_000_000)
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
    if device == "cuda":
        # Spec SS9.2 hygiene rule (inverse of zda_grid.py's default, on
        # purpose): TF32's 10-bit mantissa quantizes near-manifold
        # distance like fp16 storage does. No CLI escape hatch.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("[hygiene] TF32 disabled (spec SS9.2)")
    if device == "cpu" and not args.debug:
        print("WARNING: no GPU detected — the real grid needs one. "
              "K3 wall-clock ran 1.17-1.46x dense at CPU smoke scale; "
              "use --debug for a local plumbing test.")

    if args.debug:
        ds = SyntheticDataset()
        args.batch_size = 4
        steps_dbg = 16
        args.budget = steps_dbg * args.batch_size * DEBUG_CTX
        args.eval_tokens = 4 * args.batch_size * DEBUG_CTX
        args.eval_iters = 2
        args.out_root = args.out_root + "_debug"
        if args.variants == ",".join(GRID_ORDER):
            args.variants = "S,D0p"
        if args.seeds == "1337,1338,1339":
            args.seeds = "1337"
    else:
        meta = prepare_tinystories(args.data_dir, args.bpe_docs,
                                   args.max_train_tokens, args.max_val_tokens)
        ds = TokenDataset(args.data_dir, meta)
        assert ds.vocab_size == 4096, f"vocab {ds.vocab_size} != 4096"
        print(match_table(ds.vocab_size))

    if not args.skip_checks:
        preflight(ds.vocab_size, debug=args.debug)

    ctx = DEBUG_CTX if args.debug else GRID_CTX
    tokens_per_step = args.batch_size * ctx
    tr = {"steps": math.ceil(args.budget / tokens_per_step),
          "batch_size": args.batch_size,
          "lr": 3e-4, "lr_min": 3e-5, "warmup_steps": 2000,
          "weight_decay": 0.1, "grad_clip": 1.0,
          "eval_interval": max(1, round(args.eval_tokens / tokens_per_step)),
          "eval_iters": args.eval_iters, "log_interval": 50}
    if args.debug:
        tr["log_interval"] = 1
        tr["warmup_steps"] = 4
    print(f"\n{tr['steps']:,} steps/run x {tokens_per_step:,} tokens/step = "
          f"{tr['steps'] * tokens_per_step / 1e6:.1f}M tokens; "
          f"eval every {tr['eval_interval']:,} steps")

    variants = []
    for v in args.variants.split(","):
        key = v.strip().lower()
        assert key in _CANON, f"unknown variant {v!r} (have {VARIANTS})"
        variants.append(_CANON[key])
    seeds = [int(s) for s in args.seeds.split(",")]

    os.makedirs(args.out_root, exist_ok=True)
    for seed in seeds:
        for v in variants:
            run_dir = os.path.join(args.out_root, f"{v}_seed{seed}")
            if os.path.exists(os.path.join(run_dir, "summary.json")):
                print(f"=== {v} seed {seed}: already complete, skipping ===")
                continue
            print(f"=== {v} seed {seed} ===")
            cfg = grid_config(v, seed, ds.vocab_size, debug=args.debug)
            train_one(cfg, ds, tr, run_dir, device,
                      sim_preempt_step=args.sim_preempt_step)

    grand_summary(args.out_root, variants, seeds)


if __name__ == "__main__":
    main()
