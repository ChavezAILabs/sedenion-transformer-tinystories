"""phase4_matched_N_trained.py -- matched-N random-key null control,
rebuilt on REAL TinyStories val batches and (optionally) TRAINED
checkpoints, per the chat-side 2026-07-28 spec that superseded
phase4_matched_N_null.py's synthetic-input version.

Supersedes phase4_matched_N_null.py's methodology (that script used
torch.randn as input, RESULTS_phase4.md SS12.4 -- not comparable to
trained weights, which are coupled to the real data distribution they
were optimized on). Both `--mode init` and `--mode trained` here use the
IDENTICAL real-data pipeline so the two are directly comparable; `init`
supersedes SS12.2/SS12.4's numbers for that reason.

Differences from phase4_matched_N_null.py, each corresponding to one item
in the chat-side spec:
  1. Input is real TinyStories val-set batches (TokenDataset over
     p4_val_data/val.bin, the same class phase4_grid.py trains against),
     not torch.randn. Both init and trained modes use this.
  2. Norm-matched null: verified analytically AND numerically that the
     model's r2 = ||P(x)Q||^2/(|P|^2|Q|^2) is already exactly scale-
     invariant in ||Q|| (and ||P||) -- drawing null keys as unit vectors
     introduces no bias regardless of real-key norm drift. Real per-
     variant key-norm distributions are still logged (a finding in their
     own right, independent of whether it biases r2).
  3. Results are stratified by N-quartile, not pooled.
  4. Trajectory (early/mid/late checkpoints): NOT AVAILABLE -- checked
     directly, phase4_grid.py's checkpoint save is a single rolling
     ckpt_last.pt overwritten at every eval; only final-state ckpt.pt
     exists anywhere on Drive for S/X. Not fabricated; flagged, not
     silently dropped.
  5. Empirical percentile readout: for each point, where the actual min
     r2 falls within its own matched-N null distribution (uniform on
     [0,1] under no steering). Both ratio conventions from SS12.4(iii)
     (ratio_of_medians, median_of_ratios) kept alongside.

Focuses on LAYER 0 (matches every other headline number in this project
-- RESULTS_phase4.md's r2_min@0/@end table, PHASE5_stage0_findings SS1's
early-lock trajectory, most of H4c's descending heads).

Decision rule (pre-registered, chat-side 2026-07-28, unchanged after
seeing data):
  both ratio_null ~= 1   -> neither steers; no behavioral difference
  S < 1, X ~= 1           -> S steers, X does not; "doesn't" is supported
  both < 1                -> both steer; compare magnitudes
"""
import argparse
import sys
sys.path.insert(0, r"C:\dev\projects\apm-agi_tests")
import json

import numpy as np
import torch

from phase4_model import Phase4Model
from phase4_layers import shuffled_structure_tensor
from sedenion_kernel import structure_tensor

DATA_DIR = r"C:\dev\projects\apm-agi_tests\p4_val_data"
CKPT_DIR = r"C:\dev\projects\apm-agi_tests\p4_checkpoints"
N_ALG = 16
N_BATCHES = 5
POINTS_PER_BATCH = 40
N_RANDOM_TRIALS = 100
EPS = 1e-12


# ----------------------------------------------------------------------
# real data
# ----------------------------------------------------------------------
class TokenDataset:
    """Minimal re-implementation of phase4_grid.TokenDataset's val-only
    path (identical get_batch semantics -- seeded contiguous chunks over
    a memmapped uint16 bin)."""

    def __init__(self, data_dir):
        with open(f"{data_dir}/meta.json") as f:
            meta = json.load(f)
        self.vocab_size = meta["vocab_size"]
        self.val = np.memmap(f"{data_dir}/val.bin", dtype=np.uint16, mode="r")

    def get_batch(self, batch_size, ctx, generator):
        ix = torch.randint(len(self.val) - ctx - 1, (batch_size,),
                           generator=generator).tolist()
        x = torch.from_numpy(np.stack(
            [self.val[i:i + ctx].astype(np.int64) for i in ix]))
        return x


# ----------------------------------------------------------------------
# r2 / floor (identical math to phase4_matched_N_null.py, re-verified)
# ----------------------------------------------------------------------
def r2_of(T, P, Qs):
    A = np.einsum('mij,i->mj', T, P)
    prod = np.einsum('mj,nj->nm', A, Qs)
    num = (prod ** 2).sum(-1)
    den = (P ** 2).sum() * (Qs ** 2).sum(-1) + EPS
    return num / den


def achievable_floor(T, P):
    A = np.einsum('mij,i->mj', T, P)
    M = (A.T @ A) / ((P ** 2).sum() + EPS)
    return float(np.linalg.eigvalsh(M).min())


def verify_scale_invariance():
    """Item 2: confirm r2_of is exactly scale-invariant in ||Q|| (and
    ||P||) before deciding whether the null needs norm-matching."""
    rng = np.random.default_rng(0)
    T = structure_tensor(16)
    max_rel_diff = 0.0
    for _ in range(200):
        P = rng.normal(size=16)
        Q = rng.normal(size=16)
        r2_a = r2_of(T, P, Q[None, :])[0]
        for scale in (0.01, 0.1, 10.0, 100.0):
            r2_b = r2_of(T, P, (Q * scale)[None, :])[0]
            max_rel_diff = max(max_rel_diff, abs(r2_a - r2_b) / (r2_a + EPS))
    return max_rel_diff


# ----------------------------------------------------------------------
# model construction / loading
# ----------------------------------------------------------------------
def build_model(variant, seed, cfg, ckpt_path=None):
    """Caller is responsible for torch.manual_seed() before calling (for
    `init` mode, a fresh seed per batch; for `trained` mode it doesn't
    matter since load_state_dict overwrites every weight). `seed` here is
    only the RUN seed, used for X's shuffled-tensor construction -- for
    `trained` mode that buffer gets overwritten by load_state_dict too,
    but must still match so the module shapes/identity are right."""
    model = Phase4Model(variant, cfg["vocab_size"], cfg["d_model"],
                        cfg["n_heads"], cfg["n_layers"], cfg["mlp_hidden"],
                        seed=seed)
    if ckpt_path is not None:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ck["variant"] == variant
        model.load_state_dict(ck["model"])
    model.eval()
    return model


# ----------------------------------------------------------------------
# main per-(variant, seed, mode) run
# ----------------------------------------------------------------------
def run_condition(variant, seed, mode, cfg, ds):
    ckpt_path = (f"{CKPT_DIR}/{variant}_seed{seed}/ckpt.pt"
                if mode == "trained" else None)
    T = structure_tensor(16) if variant == "S" else shuffled_structure_tensor(seed)

    actual_mins, floors, null_mins, ratios, percentiles = [], [], [], [], []
    Ns, real_key_norms = [], []

    for batch_seed in range(N_BATCHES):
        if mode == "init":
            torch.manual_seed(1000 + batch_seed)   # fresh init per batch, as SS12.2/SS12.4
            model = build_model(variant, seed, cfg, ckpt_path=None)
        else:
            if batch_seed == 0:
                model = build_model(variant, seed, cfg, ckpt_path=ckpt_path)
        gen = torch.Generator().manual_seed(seed * 100_000 + batch_seed)
        x_tokens = ds.get_batch(cfg["batch_size"], cfg["ctx"], gen)

        with torch.no_grad():
            h = model.emb(x_tokens)
            xin = model.blocks[0].ln1(h)
            # rotated q,k for the layer-0 K3Attention, matching exactly
            # what K3Attention.scores() computes internally
            xf = xin.float()
            attn = model.blocks[0].attn
            q = attn.wq(xf).view(xf.shape[0], xf.shape[1], cfg["n_heads"],
                                 N_ALG).transpose(1, 2)
            k = attn.wk(xf).view(xf.shape[0], xf.shape[1], cfg["n_heads"],
                                 N_ALG).transpose(1, 2)
            pos = torch.arange(cfg["ctx"], dtype=torch.float32)
            q_rot = attn._rotate(q, pos).numpy()
            k_rot = attn._rotate(k, pos).numpy()

        rng = np.random.default_rng(seed * 1000 + batch_seed)
        b_idx = rng.integers(0, cfg["batch_size"], POINTS_PER_BATCH)
        h_idx = rng.integers(0, cfg["n_heads"], POINTS_PER_BATCH)
        t_idx = rng.integers(0, cfg["ctx"], POINTS_PER_BATCH)

        for bb, hh, tt in zip(b_idx, h_idx, t_idx):
            P = q_rot[bb, hh, tt, :]
            N = int(tt) + 1
            real_keys = k_rot[bb, hh, :N, :]
            real_key_norms.extend(np.linalg.norm(real_keys, axis=-1).tolist())

            actual_r2 = r2_of(T, P, real_keys)
            actual_min = float(actual_r2.min())
            floor = achievable_floor(T, P)

            trial_mins = np.empty(N_RANDOM_TRIALS)
            for trial in range(N_RANDOM_TRIALS):
                rand_Q = rng.normal(size=(N, N_ALG))
                rand_Q /= np.linalg.norm(rand_Q, axis=-1, keepdims=True) + EPS
                trial_mins[trial] = r2_of(T, P, rand_Q).min()
            random_N_min = float(np.median(trial_mins))
            percentile = float((trial_mins <= actual_min).mean())

            actual_mins.append(actual_min)
            floors.append(floor)
            null_mins.append(random_N_min)
            ratios.append(actual_min / (random_N_min + EPS))
            percentiles.append(percentile)
            Ns.append(N)

    return {
        "variant": variant, "seed": seed, "mode": mode,
        "actual_mins": np.array(actual_mins),
        "floors": np.array(floors),
        "null_mins": np.array(null_mins),
        "ratios": np.array(ratios),
        "percentiles": np.array(percentiles),
        "Ns": np.array(Ns),
        "key_norms": np.array(real_key_norms),
    }


def summarize(res, label):
    Ns = res["Ns"]
    ratios = res["ratios"]
    percentiles = res["percentiles"]
    print(f"=== {label} ===")
    print(f"  n points: {len(ratios)}  N range {Ns.min()}-{Ns.max()}")
    print(f"  key norms: median={np.median(res['key_norms']):.4f} "
          f"IQR=[{np.percentile(res['key_norms'],25):.4f}, "
          f"{np.percentile(res['key_norms'],75):.4f}]")
    print(f"  POOLED ratio_of_medians={np.median(res['actual_mins'])/np.median(res['null_mins']):.4f} "
          f"median_of_ratios={np.median(ratios):.4f} "
          f"mean_percentile={np.mean(percentiles):.4f}")
    quartile_edges = np.percentile(Ns, [25, 50, 75])
    bins = np.digitize(Ns, quartile_edges)
    for qb in range(4):
        mask = bins == qb
        if mask.sum() == 0:
            continue
        print(f"  N-quartile {qb+1} (n={mask.sum():3d}, "
              f"N in [{Ns[mask].min()},{Ns[mask].max()}]): "
              f"ratio_of_medians={np.median(res['actual_mins'][mask])/np.median(res['null_mins'][mask]):.4f} "
              f"median_of_ratios={np.median(ratios[mask]):.4f} "
              f"mean_percentile={np.mean(percentiles[mask]):.4f}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="init,trained")
    args = ap.parse_args()
    modes = args.modes.split(",")

    print("=== Item 2: r2 scale-invariance verification ===")
    max_diff = verify_scale_invariance()
    print(f"  max relative diff under ||Q|| rescaling (0.01x-100x): "
          f"{max_diff:.3e}\n")

    ds = TokenDataset(DATA_DIR)
    cfg = {"vocab_size": 4096, "d_model": 384, "n_heads": 6, "n_layers": 6,
          "mlp_hidden": 1824, "ctx": 256, "batch_size": 64}

    all_results = []
    for mode in modes:
        for variant in ("S", "X"):
            for seed in (1337, 1338, 1339):
                res = run_condition(variant, seed, mode, cfg, ds)
                summarize(res, f"{mode} {variant} seed={seed}")
                all_results.append(res)

    np.savez(r"C:\dev\projects\apm-agi_tests\p4_matched_N_trained_results.npz",
             **{f"{r['mode']}_{r['variant']}_{r['seed']}_{k}": v
                for r in all_results for k, v in r.items()
                if isinstance(v, np.ndarray)})
    print("Saved raw arrays to p4_matched_N_trained_results.npz")
