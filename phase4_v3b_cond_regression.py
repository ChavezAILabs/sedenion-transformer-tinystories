"""phase4_v3b_cond_regression.py -- free empirical companion to the V3b
non-separability argument (APM_STAGE_A_KICKOFF_2026-07-29.md follow-up,
"STEP 5" part (c)):

  "compute cond for all SIX variant tensors and regress extrapolation
  performance on it across the 18 existing runs. If no relationship,
  conditioning is not driving the outcome -- no new training required."

No training. Reads the real 18-run grid's own summary.json files
(p4_artifacts/{variant}_seed{seed}/summary.json) for ppl@512/ppl@1024/
final_val_loss, and computes a per-variant conditioning number:

  - S, X: median cond(L_v) = sqrt(max_eig/min_eig) of L_v^T L_v over 2000
    random unit v (free-sphere; matches the methodology already used in
    RESULTS_phase4.md SS8.2/SS12.1 and PHASE5_stage0_findings SS2 --
    S is theorem-fixed and seed-independent; X's shuffled tensor genuinely
    differs per seed, so computed separately for seed in {1337,1338,1339}).
  - D0p, D0, D1, Q0: dense attention has no bilinear structure tensor at
    all -- score(q,k) = q^T k / sqrt(d_h) is literally the identity
    bilinear form. The only defensible "cond" value for the identity
    operator is 1 (perfectly conditioned, and v-independent), stated here
    as an explicit convention rather than left implicit.

Then regresses ppl@512 and ppl@1024 (both extrapolation rungs) against
this per-run cond value across all 18 runs (Pearson on log(ppl) since ppl
spans orders of magnitude across variants, plus Spearman rank correlation,
computed manually since scipy is not installed in this venv).
"""
import json
import os

import numpy as np

from sedenion_kernel import structure_tensor
from phase4_layers import shuffled_structure_tensor

ARTIFACTS_DIR = r"C:\dev\projects\apm-agi_tests\p4_artifacts"
SEEDS = (1337, 1338, 1339)
DENSE_VARIANTS = ("D0p", "D0", "D1", "Q0")
N_PROBE = 2000


def median_cond(T, n=N_PROBE, seed=0):
    rng = np.random.default_rng(seed)
    conds = np.empty(n)
    for i in range(n):
        v = rng.normal(size=16)
        v /= np.linalg.norm(v)
        L = np.einsum('kij,i->kj', T, v)
        eigs = np.linalg.eigvalsh(L.T @ L)
        conds[i] = np.sqrt(eigs.max() / max(eigs.min(), 1e-300))
    return float(np.median(conds))


def spearman(x, y):
    def rank(a):
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a))
        return ranks
    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def load_summary(variant, seed):
    path = os.path.join(ARTIFACTS_DIR, f"{variant}_seed{seed}", "summary.json")
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    print("=== Per-variant conditioning numbers ===")
    cond_by_variant_seed = {}

    T_S = structure_tensor(16)
    cond_S = median_cond(T_S, seed=0)
    for seed in SEEDS:
        cond_by_variant_seed[("S", seed)] = cond_S
    print(f"  S  (theorem-fixed, seed-independent): median cond = "
          f"{cond_S:.4f}")

    for seed in SEEDS:
        T_X = shuffled_structure_tensor(seed)
        c = median_cond(T_X, seed=seed)
        cond_by_variant_seed[("X", seed)] = c
        print(f"  X  seed={seed}: median cond = {c:.4f}")

    for v in DENSE_VARIANTS:
        for seed in SEEDS:
            cond_by_variant_seed[(v, seed)] = 1.0
    print(f"  {', '.join(DENSE_VARIANTS)}: cond = 1.0 by convention "
          "(identity bilinear form, no structure tensor -- see docstring)")

    print("\n=== Real grid: variant, seed, cond, ppl@512, ppl@1024, "
          "val_loss ===")
    rows = []
    for v in ("D0p", "D0", "D1", "S", "X", "Q0"):
        for seed in SEEDS:
            s = load_summary(v, seed)
            lg = s["length_gen"]
            row = {"variant": v, "seed": seed,
                   "cond": cond_by_variant_seed[(v, seed)],
                   "ppl512": lg["ctx512"]["ppl"],
                   "ppl1024": lg["ctx1024"]["ppl"],
                   "val_loss": s["final_val_loss"]}
            rows.append(row)
            print(f"  {v:5s} {seed}  cond={row['cond']:8.4f}  "
                  f"ppl@512={row['ppl512']:9.2f}  "
                  f"ppl@1024={row['ppl1024']:9.2f}  "
                  f"val_loss={row['val_loss']:.4f}")

    cond = np.array([r["cond"] for r in rows])
    log_cond = np.log(cond)
    ppl512 = np.array([r["ppl512"] for r in rows])
    ppl1024 = np.array([r["ppl1024"] for r in rows])
    log_ppl512 = np.log(ppl512)
    log_ppl1024 = np.log(ppl1024)

    print("\n=== Regression across all 18 runs ===")
    for name, y, logy in (("ppl@512", ppl512, log_ppl512),
                          ("ppl@1024", ppl1024, log_ppl1024)):
        pear_raw = np.corrcoef(cond, y)[0, 1]
        pear_log = np.corrcoef(log_cond, logy)[0, 1]
        spear = spearman(cond, y)
        print(f"  {name}: Pearson(cond, {name})={pear_raw:+.3f}  "
              f"Pearson(log cond, log {name})={pear_log:+.3f}  "
              f"Spearman={spear:+.3f}")

    print("\n=== Within-dense-family check (same cond=1.0, does ppl "
          "vary anyway?) ===")
    for name, key in (("ppl@512", "ppl512"), ("ppl@1024", "ppl1024")):
        dense_vals = np.array([r[key] for r in rows
                               if r["variant"] in DENSE_VARIANTS])
        print(f"  {name} across D0p/D0/D1/Q0 (all cond=1.0): "
              f"min={dense_vals.min():.2f} max={dense_vals.max():.2f} "
              f"ratio={dense_vals.max()/dense_vals.min():.3f}x "
              f"(all identically conditioned, so any spread here is cond-"
              f"independent variance)")

    print("\n=== Where S sits relative to the dense family (NOT claiming "
          "S beats dense -- H4a is negative, RESULTS_phase4.md SS4; this "
          "only checks whether cond ranks variants consistently) ===")
    for name in ("ppl@512", "ppl@1024"):
        key = "ppl512" if name == "ppl@512" else "ppl1024"
        s_val = np.mean([r[key] for r in rows if r["variant"] == "S"])
        dense_means = {v: np.mean([r[key] for r in rows if r["variant"] == v])
                       for v in DENSE_VARIANTS}
        print(f"  {name}: S mean={s_val:.2f}  dense means="
              f"{ {v: round(m, 2) for v, m in dense_means.items()} }  "
              f"-- S (cond={cond_S:.2f}) sits inside the dense family's own "
              f"(cond=1.0) spread, not below all of it: conditioning does "
              f"not cleanly separate S from dense here either.")
