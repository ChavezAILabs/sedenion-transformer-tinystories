"""phase4_stageA_decision.py -- apply the pre-registered Stage A decision
rule (RESULTS_phase4.md SS12.7, locked BEFORE this script was run or its
output examined) to p4_matched_N_trained_results_rungs_512_1024.npz.

Statistic (as pre-registered): pool the 3 seeds' corr-matched-null
percentiles per (variant, rung) -- 600 points -- and compute the median
percentile and z = (0.5 - mean_percentile) / (std_percentile / sqrt(n))
against Uniform(0,1), using the empirical std of the pooled percentiles.

Per-(variant,rung) call:
  HOLDS      : pooled z <= -3 AND each of the 3 seeds individually has its
               own median percentile < 0.5
  COLLAPSES  : pooled |z| < 2
  AMBIGUOUS  : 2 <= |z| < 3 (treated as COLLAPSES for the decision rule)

3-way rule: mechanism paper iff S HOLDS at both rungs AND X is
COLLAPSES-or-AMBIGUOUS at both rungs; anything else (including a
rung-split) is the fallback paper.
"""
import numpy as np

SEEDS = (1337, 1338, 1339)
RUNGS = (512, 1024)

data = np.load(r"C:\dev\projects\apm-agi_tests\p4_matched_N_trained_results_rungs_512_1024.npz")


def classify(z):
    if z <= -3:
        return "HOLDS"
    if abs(z) < 2:
        return "COLLAPSES"
    return "AMBIGUOUS"


print(f"{'variant':8s} {'rung':6s} {'pooled median pct':>18s} {'pooled z':>10s} "
      f"{'per-seed median pct':>28s} {'call':>10s}")
results = {}
for variant in ("S", "X"):
    for rung in RUNGS:
        pooled = []
        per_seed_medians = []
        for seed in SEEDS:
            key = f"trained_{variant}_{seed}_ctx{rung}_corr_percentiles"
            arr = data[key]
            pooled.append(arr)
            per_seed_medians.append(float(np.median(arr)))
        pooled = np.concatenate(pooled)
        n = len(pooled)
        mean_pct = pooled.mean()
        std_pct = pooled.std(ddof=1)
        # NOTE: SS12.7's written formula had the subtraction order backwards
        # (0.5 - mean_pct) relative to the sign convention actually used
        # throughout SS12.5 (z negative == real steering, e.g. z~=-6.4 for
        # trained S in-distribution). Fixed here to (mean_pct - 0.5) to
        # match that established, unambiguous convention -- see the
        # transparency note in RESULTS_phase4.md SS12.9 for the full
        # account of this catch (a subtraction-order typo, not a change
        # to the pre-registered thresholds or their intended meaning).
        z = (mean_pct - 0.5) / (std_pct / np.sqrt(n))
        median_pct = float(np.median(pooled))
        c = classify(z)
        per_seed_holds = all(m < 0.5 for m in per_seed_medians)
        if c == "HOLDS" and not per_seed_holds:
            c = "HOLDS (pooled only -- per-seed consistency FAILS)"
        results[(variant, rung)] = (median_pct, z, c, per_seed_medians)
        print(f"{variant:8s} {rung:<6d} {median_pct:18.4f} {z:10.2f} "
              f"{str([round(m,3) for m in per_seed_medians]):>28s} {c:>10s}")

print()
s_calls = [results[("S", r)][2] for r in RUNGS]
x_calls = [results[("X", r)][2] for r in RUNGS]
s_holds_both = all(c.startswith("HOLDS") and "FAILS" not in c for c in s_calls)
x_collapses_both = all(c in ("COLLAPSES", "AMBIGUOUS") for c in x_calls)

print("=== PRE-REGISTERED DECISION (RESULTS_phase4.md SS12.7) ===")
print(f"  S calls at rungs {RUNGS}: {s_calls}")
print(f"  X calls at rungs {RUNGS}: {x_calls}")
if s_holds_both and x_collapses_both:
    print("  -> MECHANISM PAPER: S holds at both rungs, X collapses/ambiguous "
          "at both. Dissociation explained by differential OOD engagement.")
else:
    print("  -> FALLBACK PAPER: pre-registered clean-dissociation condition "
          "not met.")
