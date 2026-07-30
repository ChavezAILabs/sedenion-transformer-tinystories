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

2026-07-29 amendments (chat-side review of the first trained-checkpoint
pass, RESULTS_phase4.md SS12.5), all additive -- nothing above is revised:
  a. init-mode seeding fixed: was `1000+batch_seed` (independent of the
     RUN seed), so S's "3 seeds" shared bit-identical weights and only
     varied by data batch. Now `seed*10+batch_seed` -- genuinely
     independent inits per seed.
  b. i.i.d.-unit-vector null trials raised 100->300, and a null_p5 (5th
     percentile of the null draws, not just the median) is now tracked
     per point, giving a floor-free ratio (actual_min/null_p5) that
     doesn't saturate the way percentile-vs-100-trials does at extreme
     points.
  c. A second, CORRELATION-MATCHED null added alongside the i.i.d. one:
     instead of N independent random unit vectors, draw N real keys from
     an unrelated (batch, offset) window of the same length -- a real,
     naturally-correlated block of keys with no relationship to the
     query's own true keys. Under the correlation-matched null,
     percentile 0.5 IS the correct no-steering reference (unlike the
     i.i.d. null, whose reference point is shifted above 0.5 by the
     positive correlation among real same-sequence keys -- see
     RESULTS_phase4.md SS12.5 for the argument and the empirical check).

2026-07-29, APM Stage A (APM_STAGE_A_KICKOFF_2026-07-29.md STEP 3), additive:
  `run_condition` now takes an explicit `ctx` (defaults to cfg["ctx"]=256,
  identical to every prior invocation -- exact backward compatibility, same
  seed formulas, nothing renumbered). `--rungs` (default "256") lets the
  same validated correlation-matched-null instrument run on the
  EXTRAPOLATION-RUNG inputs (ctx=512/1024, same construction as
  phase4_grid.py's length_gen_eval) instead of the in-distribution val set,
  against the SAME trained S/X checkpoints -- no new training, no new grid.
  Both variants steer in-distribution (SS12.5); the S-vs-X dissociation is
  an out-of-distribution phenomenon, so this measures steering out of
  distribution directly. Per STEP 3's sanity-gate requirement, init-mode
  results at a given rung print (and its corr-matched null's median
  percentile, expected ~0.5) BEFORE that rung's trained results, so a
  failure to transfer is visible before any trained number is read.
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
N_RANDOM_TRIALS = 300
N_CORR_TRIALS = 200
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
def run_condition(variant, seed, mode, cfg, ds, ctx=None):
    """`ctx` defaults to cfg["ctx"] (256, the in-distribution val-set
    length used by every invocation before 2026-07-29). Passing a longer
    ctx (512/1024) evaluates the identical validated instrument against
    extrapolation-rung inputs -- same seed formulas below, so ctx=256
    reproduces the original run bit-for-bit; only the get_batch/pos-arange/
    point-sampling bound changes for other ctx values."""
    ctx = cfg["ctx"] if ctx is None else ctx
    ckpt_path = (f"{CKPT_DIR}/{variant}_seed{seed}/ckpt.pt"
                if mode == "trained" else None)
    T = structure_tensor(16) if variant == "S" else shuffled_structure_tensor(seed)

    (actual_mins, floors, null_mins, ratios, percentiles, ratios_p5,
     corr_mins, corr_ratios, corr_percentiles) = ([], [], [], [], [], [],
                                                   [], [], [])
    Ns, real_key_norms = [], []

    # Phase 1: compute (q_rot, k_rot) for all N_BATCHES batches up front,
    # so the correlation-matched null (phase 2) can draw "unrelated
    # window" foils from ANY of the 5 batches, not just already-seen ones.
    batch_q_rot, batch_k_rot = {}, {}
    for batch_seed in range(N_BATCHES):
        if mode == "init":
            # seed depends on the RUN seed too (not just batch_seed) so
            # each of the 3 "seeds" is a genuinely independent random
            # init, not 3 real-data replicates against one fixed init --
            # caught on review (RESULTS_phase4.md SS12.5 caveat) for S,
            # which (unlike X) has no other seed-dependence to fall back on
            torch.manual_seed(seed * 10 + batch_seed)
            model = build_model(variant, seed, cfg, ckpt_path=None)
        else:
            if batch_seed == 0:
                model = build_model(variant, seed, cfg, ckpt_path=ckpt_path)
        gen = torch.Generator().manual_seed(seed * 100_000 + batch_seed)
        x_tokens = ds.get_batch(cfg["batch_size"], ctx, gen)

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
            pos = torch.arange(ctx, dtype=torch.float32)
            batch_q_rot[batch_seed] = attn._rotate(q, pos).numpy()
            batch_k_rot[batch_seed] = attn._rotate(k, pos).numpy()

    # Phase 2: sample points per batch and compute actual/floor/both nulls.
    for batch_seed in range(N_BATCHES):
        q_rot, k_rot = batch_q_rot[batch_seed], batch_k_rot[batch_seed]
        rng = np.random.default_rng(seed * 1000 + batch_seed)
        b_idx = rng.integers(0, cfg["batch_size"], POINTS_PER_BATCH)
        h_idx = rng.integers(0, cfg["n_heads"], POINTS_PER_BATCH)
        t_idx = rng.integers(0, ctx, POINTS_PER_BATCH)

        for bb, hh, tt in zip(b_idx, h_idx, t_idx):
            P = q_rot[bb, hh, tt, :]
            N = int(tt) + 1
            real_keys = k_rot[bb, hh, :N, :]
            real_key_norms.extend(np.linalg.norm(real_keys, axis=-1).tolist())

            actual_r2 = r2_of(T, P, real_keys)
            actual_min = float(actual_r2.min())
            floor = achievable_floor(T, P)

            # -- i.i.d. unit-vector null (original, target-geometry) --
            trial_mins = np.empty(N_RANDOM_TRIALS)
            for trial in range(N_RANDOM_TRIALS):
                rand_Q = rng.normal(size=(N, N_ALG))
                rand_Q /= np.linalg.norm(rand_Q, axis=-1, keepdims=True) + EPS
                trial_mins[trial] = r2_of(T, P, rand_Q).min()
            random_N_min = float(np.median(trial_mins))
            null_p5 = float(np.percentile(trial_mins, 5))
            percentile = float((trial_mins <= actual_min).mean())

            # -- correlation-matched null (real, unrelated windows) --
            corr_trial_mins = np.empty(N_CORR_TRIALS)
            other_bs = [x for x in range(cfg["batch_size"]) if x != bb]
            for trial in range(N_CORR_TRIALS):
                b2 = other_bs[rng.integers(0, len(other_bs))]
                bs2 = rng.integers(0, N_BATCHES)   # any already-computed batch
                k_pool = batch_k_rot[bs2]
                max_off = cfg["ctx"] - N
                off = rng.integers(0, max_off + 1) if max_off > 0 else 0
                foil_keys = k_pool[b2, hh, off:off + N, :]
                corr_trial_mins[trial] = r2_of(T, P, foil_keys).min()
            corr_null_med = float(np.median(corr_trial_mins))
            corr_percentile = float((corr_trial_mins <= actual_min).mean())

            actual_mins.append(actual_min)
            floors.append(floor)
            null_mins.append(random_N_min)
            ratios.append(actual_min / (random_N_min + EPS))
            ratios_p5.append(actual_min / (null_p5 + EPS))
            percentiles.append(percentile)
            corr_mins.append(corr_null_med)
            corr_ratios.append(actual_min / (corr_null_med + EPS))
            corr_percentiles.append(corr_percentile)
            Ns.append(N)

    return {
        "variant": variant, "seed": seed, "mode": mode, "ctx": ctx,
        "actual_mins": np.array(actual_mins),
        "floors": np.array(floors),
        "null_mins": np.array(null_mins),
        "ratios": np.array(ratios),
        "ratios_p5": np.array(ratios_p5),
        "percentiles": np.array(percentiles),
        "corr_null_mins": np.array(corr_mins),
        "corr_ratios": np.array(corr_ratios),
        "corr_percentiles": np.array(corr_percentiles),
        "Ns": np.array(Ns),
        "key_norms": np.array(real_key_norms),
    }


def summarize(res, label):
    Ns = res["Ns"]
    ratios = res["ratios"]
    percentiles = res["percentiles"]
    ratios_p5 = res["ratios_p5"]
    corr_ratios = res["corr_ratios"]
    corr_percentiles = res["corr_percentiles"]
    print(f"=== {label} ===")
    print(f"  n points: {len(ratios)}  N range {Ns.min()}-{Ns.max()}")
    print(f"  key norms: median={np.median(res['key_norms']):.4f} "
          f"IQR=[{np.percentile(res['key_norms'],25):.4f}, "
          f"{np.percentile(res['key_norms'],75):.4f}]")
    print(f"  IID NULL   ratio_of_medians={np.median(res['actual_mins'])/np.median(res['null_mins']):.4f} "
          f"median_of_ratios={np.median(ratios):.4f} "
          f"median_ratio_p5={np.median(ratios_p5):.4f} "
          f"mean_pct={np.mean(percentiles):.4f} median_pct={np.median(percentiles):.4f}")
    print(f"  CORR NULL  ratio_of_medians={np.median(res['actual_mins'])/np.median(res['corr_null_mins']):.4f} "
          f"median_of_ratios={np.median(corr_ratios):.4f} "
          f"mean_pct={np.mean(corr_percentiles):.4f} median_pct={np.median(corr_percentiles):.4f}")
    quartile_edges = np.percentile(Ns, [25, 50, 75])
    bins = np.digitize(Ns, quartile_edges)
    for qb in range(4):
        mask = bins == qb
        if mask.sum() == 0:
            continue
        print(f"  Nq{qb+1} (n={mask.sum():3d}, N=[{Ns[mask].min()},{Ns[mask].max()}]): "
              f"iid_ratio={np.median(ratios[mask]):.4f} iid_pct={np.median(percentiles[mask]):.4f}  |  "
              f"corr_ratio={np.median(corr_ratios[mask]):.4f} corr_pct={np.median(corr_percentiles[mask]):.4f}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="init,trained")
    ap.add_argument("--rungs", default="256",
                    help="context lengths to test. Default 256 = the "
                         "in-distribution val set, identical seed formulas "
                         "to every run before 2026-07-29 (exact "
                         "reproduction). Pass 512,1024 for APM Stage A: "
                         "the extrapolation rungs, same trained checkpoints, "
                         "no new training (APM_STAGE_A_KICKOFF_2026-07-29.md "
                         "STEP 3).")
    args = ap.parse_args()
    modes = args.modes.split(",")
    rungs = [int(r) for r in args.rungs.split(",")]

    print("=== Item 2: r2 scale-invariance verification ===")
    max_diff = verify_scale_invariance()
    print(f"  max relative diff under ||Q|| rescaling (0.01x-100x): "
          f"{max_diff:.3e}\n")

    ds = TokenDataset(DATA_DIR)
    cfg = {"vocab_size": 4096, "d_model": 384, "n_heads": 6, "n_layers": 6,
          "mlp_hidden": 1824, "ctx": 256, "batch_size": 64}

    all_results = []
    for rung in rungs:
        is_extra_rung = rung != cfg["ctx"]
        if is_extra_rung:
            print(f"##### RUNG ctx={rung} #####\n")
        rung_results = {}
        for mode in modes:
            for variant in ("S", "X"):
                for seed in (1337, 1338, 1339):
                    res = run_condition(variant, seed, mode, cfg, ds, ctx=rung)
                    label = f"{mode} {variant} seed={seed}"
                    if is_extra_rung:
                        label += f" rung={rung}"
                    summarize(res, label)
                    all_results.append(res)
                    rung_results[(mode, variant, seed)] = res
            # STEP 3 sanity gate: report init's transfer to this rung BEFORE
            # any trained number at this rung is trusted/read.
            if mode == "init" and is_extra_rung and "trained" in modes:
                pcts = [np.median(rung_results[("init", v, s)]["corr_percentiles"])
                        for v in ("S", "X") for s in (1337, 1338, 1339)]
                print(f"--- SANITY GATE (rung={rung}): init corr-matched "
                      f"median percentiles, S x3 seeds then X x3 seeds = "
                      f"{[f'{p:.3f}' for p in pcts]} (expect ~0.5 per "
                      f"SS12.5's validation at ctx=256; a large deviation "
                      f"here means the correlation-matched null does not "
                      f"transfer to ctx={rung} and the trained numbers "
                      f"below are NOT trustworthy as-is) ---\n")

    suffix = "" if rungs == [256] else "_rungs_" + "_".join(str(r) for r in rungs)
    out_path = (r"C:\dev\projects\apm-agi_tests\p4_matched_N_trained_results"
                f"{suffix}.npz")
    np.savez(out_path,
             **{f"{r['mode']}_{r['variant']}_{r['seed']}_ctx{r['ctx']}_{k}": v
                for r in all_results for k, v in r.items()
                if isinstance(v, np.ndarray)})
    print(f"Saved raw arrays to {out_path}")
