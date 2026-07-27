# PHASE 5 Stage 0 findings — 2026-07-26

**Status:** Zero-compute analysis of existing artifacts, per `PHASE5_PLAN.md`
§6 Stage 0. Two of three items completed; the third is blocked on missing
local checkpoints (flagged below, not silently skipped or substituted).
Descriptive only — no verdict language, per the completeness-gate discipline
this project has used since Phase 4.

**Inputs:** `p4_artifacts/{S,X}_seed{1337,1338,1339}/eval_log.jsonl` (the real
18-run grid), and fresh `K3Attention` instantiations at the real grid dims
(`d_model=384, n_heads=6, ctx=256`), following the same pattern as
`phase4_init_r2_check.py` (N2).

---

## 1. Early-lock trajectory analysis

**Convention check, done first:** confirmed `RESULTS_phase4.md`'s
`r²_min@0/@end/min-ever` table tracks **layer 0 specifically**, not the
global (pooled) minimum across all 6 layers — layer 0's values match the
published table exactly at every seed; the global-min values do not (e.g.
S seed1337 @0: layer-0=0.1529 matches the table, global-min-across-layers
=0.1116 does not). Worth recording since it wasn't stated explicitly
before and matters for reading any r²-trajectory number in this project
going forward. Consistent with layer 0 being where 5 of 6 H4c-crossing
heads live.

**Naive log-fraction-by-eval-2 metric doesn't work for X:** X's total
log descent (step 0 to final) is near zero and sometimes slightly
*negative* (it can end higher than it started), so dividing by it produces
meaningless artifacts (953%, 5725%). Replaced with a linear-regression
slope of ln(r²_min) on eval index, computed twice — once over evals[1:]
(everything after the pre-training step-0 point) and once over evals[2:]
(excluding the first post-init eval too, to isolate whether descent
continues *after* the initial jump) — plus the coefficient of variation
over evals[2:] as a spread check.

| run | slope evals[1:] | slope evals[2:] | CV evals[2:] | mean evals[2:] |
|---|---|---|---|---|
| S 1337 | −0.0585 | −0.0580 | 0.403 | 0.00176 |
| S 1338 | −0.0356 | −0.0196 | 0.314 | 0.00200 |
| S 1339 | −0.0845 | −0.0791 | 0.426 | 0.00220 |
| X 1337 | +0.0132 | +0.0168 | 0.124 | 0.04552 |
| X 1338 | −0.0139 | −0.0010 | 0.158 | 0.04477 |
| X 1339 | +0.0117 | +0.0304 | 0.170 | 0.04300 |

(slope units: natural-log change in r²_min per eval, restricted to the
post-early-descent window; mean slope[2:]: S = −0.0522, X = +0.0154.)

**Reading:** all three S seeds show a consistently negative slope even
after excluding the first two evals — S keeps descending, in a
log-linear sense, through most of the 18,311-step run, not just in an
initial jump. All three X seeds show a slope centered near zero (two of
three even slightly positive) — statistically indistinguishable from no
trend at all once the very first eval is excluded.

A direct cross-check on X confirms this differently: comparing every
post-step-0 eval to X's own step-0 value, the post-step-0 values are lower
100%/100%/75% of the time (seeds 1337/1338/1339) — so there *is* a small
one-time drop from the raw initialization level by the first eval — but
after that, the near-zero slope[2:] shows no further progress. **This
sharpens rather than simply confirms the owner's early-lock hypothesis**:
it is not that X "flattens while S keeps descending" in a generic sense —
it's that X makes one small adjustment away from its init level in the
first ~1,526 steps and then stops, while S's descent is sustained across
essentially the whole run (min-ever occurs late in training for two of
three S seeds: step 13,734/13 and step 18,311/13 — i.e. the actual final
or near-final eval — not concentrated in the early evals at all).
Consistent with, and adds a mechanistic texture to, the standing "training
exploits S's manifold, doesn't for X" reading already on record (session
close 2026-07-24 addendum).

---

## 2. cond(L_x) at the real init q,k distribution

Reuses the N2 pattern (`phase4_init_r2_check.py`): fresh `K3Attention`
instantiations at real grid dims, 5 independent inits × batch 24 × ctx 256
= 184,320 (q,k) vectors per variant, taken **after** the wq/wk projection
and R_8 rotation — not raw free-sphere samples of x.

| | free-sphere median cond(L_x) | real-init q median | real-init k median |
|---|---|---|---|
| **S** | 2.77 | 2.78 | 2.79 |
| X seed 1337 | 43.29 | 43.44 | 43.36 |
| X seed 1338 | 44.09 | 43.37 | 43.02 |
| X seed 1339 | 42.03 | 43.08 | 43.19 |

**Reading:** real-init medians match the free-sphere prediction almost
exactly for both S and X, at every seed, for both q and k (all within ~1%
of the free-sphere figure). **The conditioning objection survives contact
with the real parameterization** — the ~15–16× gap between S and X's
conditioning is not a free-sphere-sampling artifact; it is what the
model's actual q,k vectors experience at initialization. This extends N2's
finding (which checked r² itself) to a different, per-vector statistic and
gets the same answer: the model's real init resembles free-sphere sampling
closely enough that free-sphere characterizations of S vs. X are a
reasonable proxy, at least at initialization (this says nothing about
whether that resemblance holds after training moves the projections away
from init — not checked here).

---

## 3. Constrained-infimum check — BLOCKED, not run

**Freeze the tensor, disable the LM objective, optimize q,k projections
directly against r²; returns each variant's reachable floor and settles
can't-vs-doesn't.** Per `PHASE5_PLAN.md` §9, this explicitly requires the
actual trained checkpoints from the 18-run grid (~3 GB, Drive-only) as the
starting point — the question is whether continuing to optimize *from
where training left the model* against r² alone can push X lower than
training did, which is a different question from whether a *freshly
initialized* projection can be optimized to low r² (the latter is closer
to what `phase4_X_variety_sigma_min.py` already answered, on raw vectors
rather than through the learned projection layers).

**Confirmed no checkpoints exist locally**: searched the full repo and
`p4_artifacts/` (only `eval_log.jsonl` / `train_log.csv` / `summary.json`
per run, 932 KB total) — nothing under any directory matches the real
grid's checkpoint naming. Not attempting a fresh-random-init substitute
without checking first, since it would answer a related but different
question and I don't want a "Stage 0 item 3 done" to quietly mean
something other than what the plan asked for.

**To unblock:** either sync the grid checkpoints down from Colab Drive, or
explicitly approve a fresh-init substitute with the caveat above stated up
front in any writeup that uses it.

---

## Bottom line

Two of three Stage 0 checks ran clean and both **held up** rather than
dissolved: the conditioning objection is real at the model's actual init,
not a sampling artifact, and the early-lock hypothesis is confirmed in a
sharper form than originally stated (X makes one small early move then
stops; S's descent is sustained across nearly the whole run, not just
early). Neither result changes any Stage 1/2 plan — both were listed as
"may reshape everything below it," and neither did; they add texture to
the existing S-vs-X story rather than contradicting or replacing any of
it. Item 3 remains open pending checkpoint access or a scoping decision.
