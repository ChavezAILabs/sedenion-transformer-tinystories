# Priority A — the crossing-point run (2026-07-31)

Executes Priority A of the 2026-07-29 APM work block: extend the ppl@512/ppl@1024
length-generalization eval to longer contexts and determine whether S's and D1's
degradation curves cross in raw perplexity. No training; inference only, on the
18 existing grid checkpoints (`MyDrive/p4_runs_ts`, pulled locally to
`p4_checkpoints/`) and the existing val split (`MyDrive/zda_data_cache`, byte-identical
to the local `p4_val_data/val.bin` used for the self-check below). Run on a Colab
A100 (local CPU was tested first and found infeasible — could not finish a single
dense-variant combo at ctx=4096 in 61 minutes; see `phase4_priorityA_crossing.py`
history for the full diagnostic trail).

## Gate 0 — positional encoding (checked before running anything)

PASS. S, X, and D1 all use the same `R_8`/`omega_h` per-head rotary-style relative
encoding (`phase4_layers.py` `K3Attention` for S/X; ladder-mode `DenseAttention`
for D1) — confirmed by reading the code, not assumed. This is a relative encoding,
so evaluating past the trained ctx=256 is meaningful (not confounded the way a
learned-absolute scheme would be).

## Protocol fidelity — self-check before trusting anything

`phase4_priorityA_crossing.py` reuses `phase4_grid.ce_loss` directly and the exact
seed formula (`cfg.seed*1000+ectx`), so ctx256/512/1024 must reproduce each
checkpoint's own recorded `p4_artifacts/*/summary.json` numbers exactly before any
new point is trusted. They do, to the digit:

| | ppl@512 (seed 1337/1338/1339) | ppl@1024 |
|---|---|---|
| S, recorded | 12.137 / 11.216 / 10.75 | 39.806 / 34.376 / 30.52 |
| S, recomputed | 12.137 / 11.216 / 10.75 | 39.806 / 34.376 / 30.52 |
| X, recorded | 18.665 / 18.045 / 17.959 | 52.219 / 51.605 / 51.096 |
| X, recomputed | 18.665 / 18.045 / 17.959 | 52.219 / 51.605 / 51.096 |

## Disclosed methodology note: per-ctx batch size

ctx≤2048 uses the original protocol unchanged (`batch_size=8, iters=32`, matching
the self-check above exactly). K3's `scores()` materializes a full
`(batch, heads, T, T, 16)` tensor (`prod = einsum("bhti,bhsim->bhtsm", q, k_rot)`)
and then a second tensor of the *same* full size (`prod.pow(2)`) simultaneously —
peak memory is ~2× that tensor's size, not 1×. At `batch=8`, this exceeds a 40GB
A100 already at ctx=4096. ctx=4096 and ctx=8192 therefore use smaller batch sizes
(2 and 1 respectively) with proportionally more iterations, so every ctx still
averages the same **N=256 examples** throughout — a chunking change for memory
only, not a change to what's measured (each example's loss is computed identically
regardless of what batch it's grouped into). Dense variants (D0/D0p/D1/Q0) have no
such factor (plain `(B,H,T,T)` attention, no extra 16×) and ran the unmodified
protocol at every length including 8192.

## Full results table (pooled mean ± std over 3 seeds, raw perplexity)

| variant | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|
| S | 6.10±0.04 | 11.37±0.71 | 34.90±4.67 | 74.92±10.97 | 117.15±17.81 | **not reachable** |
| X | 5.91±0.02 | 18.22±0.39 | 51.64±0.56 | 92.50±1.25 | 125.58±1.23 | **not reachable** |
| D1 | 5.23±0.03 | 13.35±4.23 | 34.04±8.95 | 65.10±11.18 | 89.12±9.81 | 106.13±7.19 |
| D0 | 5.23±0.01 | 7.89±0.55 | 20.69±2.83 | 42.62±6.25 | 70.86±9.94 | 99.18±14.68 |
| D0p | 5.34±0.01 | 9.84±2.84 | 28.90±5.52 | 60.55±6.79 | 87.63±6.49 | 108.44±5.95 |
| Q0 | 6.25±0.03 | 20.54±0.84 | 45.88±1.16 | 70.19±1.99 | 85.77±2.06 | 96.35±2.73 |

S and X could not be evaluated at ctx=8192 on a 40GB A100 at any batch size —
see "The ctx=8192 ceiling" below. This is a hardware/architecture limit found and
confirmed live, not a skipped measurement.

## The central question: do S and D1 cross in raw ppl?

**No — the gap between D1 and S widens monotonically across every rung from 1024
to 4096, the entire measurable range.** D1 is ahead (lower ppl) at every one of
those three rungs, by an increasing margin:

| ctx | S − D1 (pooled) | relative |
|---|---|---|
| 1024 | +0.86 | D1 better by 2.5% |
| 2048 | +9.81 | D1 better by 15.1% |
| 4096 | +28.03 | D1 better by 31.5% |

Per-seed, this strengthens rather than weakens at the longest measured rung: at
ctx=1024 the picture was mixed (1 of 3 seeds had D1 ahead, 2 had S ahead — the
"30–40% seed swings" already on record in `RESULTS_phase4.md`), but **at ctx=4096,
all three seeds show D1 ahead of S**, consistent with the pooled trend rather than
contradicting it.

(At the two shortest rungs the picture is different and not part of this claim:
D1 is ahead at ctx=256 by 16.7%, but S is actually ahead at ctx=512 by 14.8% — a
short-range wobble, not a trend, and not what "extrapolation" refers to here.)

**Conclusion for the paper: "S wins on extrapolation" cannot be stated as a raw-ppl
claim.** It has to stay on the normalized metric (ppl@1024 / e^val_loss, which
divides out each model's own baseline) exactly as `RESULTS_phase4.md` §6.1 already
flagged as a live risk. In absolute terms, D1 is not just better at ctx=1024 — it
pulls further ahead the longer the sequence gets, at least through ctx=4096, which
is as far as we can measure. We cannot rule out a crossing beyond 4096 (see below),
but there is zero evidence for one in the measured range, and the trend argues
against it, not for it.

## S vs X: no crossing, but the gap shrinks fast

S beats X in raw ppl at every measurable rung (256 is a 3.1% exception in X's
favor, noise-sized) — but the *margin* by which S wins collapses steadily as
context grows:

| ctx | X vs S | 
|---|---|
| 512 | X worse by 60.3% |
| 1024 | X worse by 48.0% |
| 2048 | X worse by 23.5% |
| 4096 | X worse by 7.2% |

At ctx=4096 this is thin enough that one of three seeds (1337) already shows **X
beating S** (125.29 vs 134.657) even though the pooled mean still favors S. This
is the first individual-seed crossing anywhere in the S-vs-X comparison. Whether
the pooled mean itself crosses beyond 4096 is unknown — both variants hit the same
hardware ceiling at 8192. This narrowing trend is real and worth stating in the
paper as an open question, not as a resolved result either way.

## Does everything degenerate together at extreme length?

Severely, but not to noise. All four dense variants degrade 15–20× from ctx=256 to
ctx=8192 (D0: 19.0×, D0p: 20.3×, D1: 20.3×, Q0: 15.4×) — a model trained at ctx=256
clearly has little usable positional structure left by ctx=8192. It is not total
collapse either: ppl plateaus around 85–115, far below the ~4096 a uniformly random
model would give. One undirected, purely descriptive observation: **Q0 (the
no-interaction floor control, never graded) has the *lowest* ppl of the four dense
variants at ctx=8192** — not interpreted here, just on the record.

## The ctx=8192 ceiling — confirmed as architectural, not a bug

Extensive live debugging on Colab (memory diagnostics inserted before/after every
context length) ruled out caching/fragmentation as the cause. The real explanation:
`K3Attention.scores()` computes `prod = einsum(...)` at shape `(B,H,T,T,16)`, then
`num = prod.pow(2).sum(-1)` — `prod` and `prod.pow(2)` are both full-size and must
coexist, so peak memory is **2×** the size of that one tensor, not 1×. At
`batch=1, ctx=8192`, that tensor alone is 24.0 GB (confirmed exactly against the
error message's "Tried to allocate 24.00 GiB"), so the peak is ~48 GB against a
39.49 GB card — and `batch=1` is already the floor, so no batch-size reduction can
fix it. Dense attention has no such factor (`(B,H,T,T)`, no extra 16×), which is
why D0/D0p/D1/Q0 ran cleanly through 8192 while S/X could not.

This is itself a legitimate, disclosable cost-side finding, distinct from but
consistent with Phase 4's already-documented 2.39× wall-clock overhead: **K3's
materialized per-head r² tensor imposes an O(batch·T²·16) memory ceiling that
dense attention's O(batch·T²) does not have.** Reaching ctx=8192 for S/X would
require chunking the computation inside `scores()` itself (e.g. tiling over the
sequence dimension) — a real change to the validated core kernel that would need
to go back through the NumPy-mirror-first validation chain
(`test_phase4_numpy.py`, the P4T1–T8 torch tests) before it's trustworthy. Not
attempted here; flagged as a possible follow-up, not started.

## Bottom line for the paper

1. **The absolute-ppl "wins with no caveat" framing does not survive contact with
   the data.** D1 stays ahead of S in raw ppl at every rung from 1024 through
   4096, and the gap widens (not narrows) with context. The claim must stay on the
   normalized metric.
2. **The S-vs-X dissociation itself is intact and, if anything, sharpened**: S
   beats X cleanly through 2048, with the margin shrinking steadily toward 4096
   (one seed already crosses there). This is a real, still-open trend, not a
   resolved crossing.
3. **A new, disclosable cost-side finding**: K3's per-head score computation has a
   hard memory ceiling — O(batch·T²·16) vs dense attention's O(batch·T²) — that
   blocks evaluation past ctx≈4096–8192 on a 40GB GPU, independent of and in
   addition to the already-documented 2.39× wall-clock cost.
