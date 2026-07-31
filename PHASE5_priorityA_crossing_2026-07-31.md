# Priority A — the crossing-point run (2026-07-31)

**REVISED 2026-07-31, same day, after chat-side review, then revised again same
day after a second round of follow-up queries against the same review.** A review
(`PHASE5_priorityA_review_2026-07-31.md`, chat-side/Claude.ai, explicitly
unverified against the repo — "reproduce from source before acting") raised five
points against the original version of this document. Four were checked directly
against the repo and confirmed correct; the fifth was checked and refuted. Nothing
was accepted on the review's arithmetic alone — every confirmed point was
independently re-derived from source (`p4_artifacts/*/summary.json` per-seed
`final_val_loss`, or a direct empirical test), not from the review's own numbers.
A follow-up round then asked for the per-seed sign counts behind two of the
pooled-mean claims below, which the first revision had not reported — those counts
changed the wording again (see the two bullets below). Audit trail, what changed
across both revisions:

- **Bottom line #1 originally recommended falling back to the normalized metric**
  as the safe claim if raw ppl failed. That recommendation was withdrawn in the
  first revision — the normalized metric also crosses D1's favor between ctx=2048
  and ctx=4096 on the pooled mean. The **second** revision downgrades this further:
  every rung from 512 through 4096 is a 2-of-3 or 1-of-3 seed split on this metric,
  not a unanimous one — only ctx=256 is seed-unanimous. "Crosses" describes the
  pooled mean; per-seed the picture is one seed (1337) favoring D1 from ctx=512
  onward, joined by a second (1338) only at ctx=4096.
- **Bottom line #2 (Q0's rank climb) is confirmed as a real ranking fact but is
  also pooled-mean-driven, not seed-unanimous**, at the level of the individual
  pairwise comparisons underneath it (Q0 vs S at ctx=2048: 2/3; Q0 vs D1 at
  ctx=4096: 2/3; Q0 vs D0p at ctx=4096: 1/3 — Q0 does not reliably beat D0p
  per-seed even where the pooled mean ranks it ahead). The interpretable-range
  convention is kept as a caution regardless (per the review's own instruction:
  "I'd keep it either way — but the memo has to say which it is").
- The originally-flagged BLOCKING batch-size pairing concern (§3) was tested and
  refuted; `PHASE5_priorityA_review_2026-07-31.md` is annotated in place marking
  it RESOLVED rather than left to read as still-live in the historical record.

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

**Restated per the review's point 4, confirmed correct.** S, X, and D1 all use the
same `R_8`/`omega_h` per-head rotary-style relative encoding — but this establishes
only that **the same code path is called for all three variants**, not that it
behaves identically. `CLAUDE.md` and `RESULTS_phase4.md` §12.5 already establish
that X's rotation does **not** preserve norms or the shared-phase null cleanly
(`phase4_X_positional_check.py`: `L_8²≠−I`, norm preservation violated, shared-phase
null broken at every head/seed for X), while S's does (Moreno's norm identity,
confirmed to 4.4×10⁻¹⁶). Gate 0 therefore passes as "extrapolation past ctx=256 is
architecturally meaningful for all three (relative, not learned-absolute,
encoding)" — it does not certify that S and X are positionally comparable to each
other, and should not be read that way.

*Speculative, labeled as such:* the S-vs-X gap **narrows** with context (below),
which argues mildly against X's positional damage being the driver of that gap — a
generator that degrades with offset would predict the opposite. Caveat: the damage
may saturate within a few positions, in which case this observation is
uninformative rather than exculpatory. Not carried forward as evidence in either
direction without a direct measurement of how X's generator error scales with
offset.

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

## Methodology note: per-ctx batch size, and the pairing check it raised

ctx≤2048 uses the original protocol unchanged (`batch_size=8, iters=32`, matching
the self-check above exactly). K3's `scores()` materializes a full
`(batch, heads, T, T, 16)` tensor and then a second tensor of the *same* full size
(`prod.pow(2)`) simultaneously — peak memory is ~2× that tensor's size. At
`batch=8`, this exceeds a 40GB A100 already at ctx=4096. ctx=4096 and ctx=8192
therefore use smaller batch sizes (2 and 1 respectively) with proportionally more
iterations, so every ctx still averages the same **N=256 examples**.

**The review's point 3 (checked, refuted):** since S/X ran ctx=4096 at
`batch=2,iters=128` while dense variants ran the unmodified `batch=8,iters=32` at
the *same* ctx, the review flagged that this might break the implicit pairing —
same generator seed (`cfg.seed*1000+ectx`), but different batch shapes might
consume the CPU generator's random stream differently, drawing different actual
windows. Tested directly: a fixed-seed `torch.Generator` drawing 32×(8,) and a
fresh same-seed generator drawing 128×(2,) produce **bit-identical** flattened
index sequences (verified locally, no GPU needed — the index draw is CPU-side
regardless of eval device). The pairing is intact at ctx=4096; dense and K3
variants really did see the same 256 windows despite the different batch
configuration. No further action needed here.

**This makes the raw-ppl finding at ctx=4096 stronger, not just unblocked**: since
the windows are confirmed identical, "all 3 seeds show D1 ahead of S at ctx=4096"
(below) is genuine paired evidence — the same 256 examples scored by both models —
not three independent draws that happened to agree. (One residual, not worth
chasing: an 8-element vs. a 2-element batch mean can differ in the last few bits of
floating-point rounding order; against a raw-ppl gap of 28, this is noise about
noise.)

## Full results table (pooled mean ± std over 3 seeds, raw perplexity)

Pooled as the arithmetic mean of ppl (matching `RESULTS_phase4.md`'s existing
convention). Checked against geometric-mean pooling (`exp(mean(ln ppl))`, the
theoretically cleaner reduction since ppl is exponential in loss): the difference
is small even at S's highest-variance point (ctx4096: 117.15 arithmetic vs 116.24
geometric, +0.8%) — real, but not large enough to change any conclusion below, so
the arithmetic-mean table is kept for consistency with the rest of this project's
tables.

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

## The central question: do S and D1 cross — on EITHER metric?

**Raw ppl: no crossing in the interpretable range (see the Q0 section below for
why "interpretable range" matters), and unlike the normalized metric below, this
one IS seed-unanimous by ctx=4096.** D1 leads on the pooled mean at every rung
from 1024 through 4096, gap widening monotonically: +2.5% (1024), +15.1% (2048),
+31.5% (4096). Per-seed win counts for D1 (of 3): **1/3 at ctx=1024** (only
seed1337; seeds 1338/1339 individually favor S despite the pooled mean favoring
D1 by 2.5% — a mean-driven pooled result at that rung), **2/3 at ctx=2048** (1337,
1338; seed1339 still favors S), **3/3 at ctx=4096** (all seeds). This is a real,
monotonically strengthening, seed-consistent trend — 1 of 3 → 2 of 3 → 3 of 3 —
not just a pooled-mean artifact, and per the methodology note above, the ctx=4096
figure is confirmed **paired** evidence (identical windows across variants).
**Statistical honesty at n=3:** a one-tailed sign test on 3/3 agreement has
p=0.125 under the null — not significant at conventional thresholds, and this
document does not claim significance for it. That is a real limit, not a reason
to treat this on par with the normalized metric above: unlike that metric, this
one is seed-consistent (monotonically 1→2→3 of 3, not a static 2/3 carried by one
idiosyncratic seed) and confirmed paired. "Strongest single result in this
document" describes its evidential weight *relative to everything else here*,
under n=3's honest ceiling — not a claim of statistical significance.

**Normalized metric (ppl@ctx / e^own-val-loss): the pooled mean also crosses, but
per-seed this is NOT the clean, later-crossing story the first revision of this
document told.** That first revision treated the normalized metric as the safe
fallback claim, found the pooled mean crosses between 2048 and 4096, and stopped
there without checking per-seed signs — an omission a follow-up review query
caught. Recomputed rigorously (normalize *each seed's own* ppl by *that same
seed's own* `e^val_loss`, then pool — not pooled ppl divided by pooled val_loss),
**with per-seed sign counts at every rung**:

| ctx | S normalized (pooled mean) | D1 normalized (pooled mean) | pooled winner | per-seed split (S wins / 3) |
|---|---|---|---|---|
| 256 | 1.011 | 1.016 | S by 0.5% | **3/3 — unanimous** |
| 512 | 1.883 | 2.594 | S by 27.4% | 2/3 (D1: seed1337 only) |
| 1024 | 5.780 | 6.617 | S by 12.7% | 2/3 (D1: seed1337 only) |
| 2048 | 12.406 | 12.655 | S by 2.0% | 2/3 (D1: seed1337 only) |
| 4096 | 19.400 | 17.322 | **D1 by 10.7%** | 1/3 (D1: seed1337, seed1338) |

**Only ctx=256 is seed-unanimous.** From ctx=512 through ctx=4096, every rung is a
2-of-3-or-1-of-3 split, never 3/3 — this is a *plurality* shift (seed1337 alone
favoring D1 from 512 onward, joined by seed1338 only at 4096), not a clean
unanimous crossing that happens to land between 2048 and 4096. Per the review's own
downgrade rule, none of ctx512/1024/2048/4096 should be read as "S ahead" or "D1
ahead" in an unqualified sense — each is "pooled mean favors [S/D1]; per-seed
evidence is split 2/1."

**Corrected conclusion, second pass: the normalized metric is the noisier of the
two, not the safer one.** Raw ppl gives a seed-consistent, monotonically
strengthening D1-ahead trend from 1024 to 4096. The normalized metric gives a
pooled-mean trend that is never seed-unanimous past ctx=256 and is carried by one
or two seeds' idiosyncrasy (seed1337 in particular) rather than a broad shift. The
honest range-bounded statement is: *on raw ppl, D1 pulls ahead of S with
increasing seed consensus from ctx=1024 to ctx=4096. On the normalized metric, the
pooled mean tells a similar late story, but it is not corroborated by seed-level
unanimity at any rung past ctx=256 — treat it as suggestive, not confirmatory.*

## Q0's rank climbs from last to first — and it starts well before ctx=8192

**The review's point 2, checked directly, confirmed correct.** Ranking all
variants by raw pooled ppl at every rung (1 = lowest ppl = best):

| ctx | ordering (best → worst) | Q0 rank |
|---|---|---|
| 256 | D1 < D0 < D0p < X < S < Q0 | 6/6 |
| 512 | D0 < D0p < S < D1 < X < Q0 | 6/6 |
| 1024 | D0 < D0p < D1 < S < Q0 < X | 5/6 |
| 2048 | D0 < D0p < D1 < **Q0** < S < X | 4/6 |
| 4096 | D0 < **Q0** < D0p < D1 < S < X | 2/6 |
| 8192 | **Q0** < D0 < D1 < D0p | 1/4 |

**Q0 — the no-interaction floor control, by construction the least capable model
in the grid, dead last at ctx=256 — climbs monotonically to first place by
ctx=8192, already 2nd of 6 by ctx=4096.** Two readings, not adjudicated here:
(1) past ~2048, ppl ordering ranks models by how much length-fragile structure
they had to lose, not by capability; (2) Q0 genuinely has better length robustness
*because* it has no query-dependent interaction to break — a real property,
orthogonal to language-modeling quality. Either way, the consequence for this
document is the same: **the headline "all three seeds show D1 ahead of S at
ctx=4096" is a true observation made in a regime where the floor control is also
ahead of D1.** It cannot be read as evidence about extrapolation *quality* at that
rung, only as a fact about raw degradation.

**Per-seed check on the two load-bearing comparisons underneath this ranking (a
follow-up review query, since a pooled-mean rank climb could itself be
mean-driven):**
- **Does Q0 beat S at ctx=2048, in all 3 seeds?** No — **2/3** (seed1337: Q0=68.06
  vs S=86.38, Q0 wins; seed1338: Q0=71.99 vs S=73.85, Q0 wins; seed1339: Q0=70.51
  vs S=64.52, **S wins**).
- **Does Q0 beat D1 and D0p at ctx=4096, in all 3 seeds each?** No to both. Q0 vs
  D1: **2/3** (seed1337: D1 wins 78.99<83.49; seeds 1338/1339: Q0 wins). Q0 vs
  D0p: **1/3** (only seed1337: Q0 wins 83.49<94.89; seeds 1338/1339: **D0p wins**,
  82.40<87.51 and 85.59<86.32).

**So the rank climb is real as a pooled-mean fact but mean-driven at the level of
the individual comparisons that produce it** — Q0 does not reliably beat any one
specific competitor per-seed at the rungs checked, even where the pooled ranking
shows it ahead. This does not overturn the convention below; the review's own
framing anticipated this ("if mean-driven, the convention still stands as a
caution — I'd keep it either way — but the memo has to say which it is"), so it is
kept, stated explicitly as a caution rather than a seed-confirmed finding.

**Adopted interpretable-range convention, going forward (a caution, not a
seed-confirmed finding — see above), with an explicit carve-out this document
initially left implicit:** this convention governs **cross-field ordering
claims** — ranking S/X/D0/D0p/D1/Q0 against each other as a group, which is
exactly what Q0's contamination undermines, since it says the six-way ordering
stops tracking capability once a no-interaction floor control starts winning.
It does **not** govern **paired within-pair comparisons on identical windows** —
a two-variant sign test (like S vs D1, above) is a different kind of claim, and a
third variant's position moving elsewhere in the field does not undermine it.
So: cross-field ordering is cost-only beyond ctx≈2048 (reliable through ctx≤1024,
reportable-with-caveat at 2048); a **paired** two-variant comparison remains
readable past that point, with the caveat that what it measures is **relative
rate of collapse**, not capability — "D1 degrades less than S from ctx=1024 to
ctx=4096" is a defensible paired-comparison statement inside the cost-only
regime; "D1 is a better model than S at ctx=4096" is not, because Q0's climb
shows ppl ordering in that regime doesn't mean what it looks like it means for
capability. This preserves both results without either silently overriding the
other — the raw-ppl S-vs-D1 finding above is not discounted by this section, and
this section is not contradicted by that finding.

## S vs X: the dissociation is intact but *decaying*, not sharpened

**Correction:** the original version of this document's bottom line described this
as "intact and, if anything, sharpened." That is wrong and contradicted its own
body text two paragraphs earlier — the margin falls monotonically, it does not
sharpen. S still wins at every measurable rung (256 is a 3.1% X-favoring
exception, noise-sized), but the *margin* collapses steadily:

| ctx | X worse than S by |
|---|---|
| 512 | 60.3% |
| 1024 | 48.0% |
| 2048 | 23.5% |
| 4096 | 7.2% |

One of three seeds (1337) already shows X beating S at ctx=4096 (125.29 vs
134.657) even though the pooled mean still favors S. Whether the pooled mean
itself crosses beyond 4096 is unknown — both variants hit the same hardware
ceiling at 8192.

**An open tension with Stage A, not resolved here.** `RESULTS_phase4.md` §12.9
(APM Stage A, pre-registered and locked) found X's out-of-distribution steering
AMBIGUOUS at ctx=512 (z=−2.91) and COLLAPSED outright at ctx=1024 (z=−0.25,
indistinguishable from no effect), supporting "only S's engagement transfers
out-of-distribution" as the mechanism-paper reading. If that is the explanation
for the behavioral gap, the gap should persist or widen past ctx=1024, where X has
no measured engagement left to lose. **Observed instead: the behavioral gap keeps
shrinking — largest at exactly the two rungs Stage A measured (512, 1024), decayed
to near-nothing at the two rungs Stage A did not measure (2048, 4096).** This is a
comparison across two independent measurements, not an attribution, and it does
not overturn Stage A's locked, pre-registered call. But it is a real tension that
belongs in the paper's mechanism section, not omitted. A possible follow-up
(inference only, no training) would extend Stage A's correlation-matched-null
instrument to ctx=2048/4096 with its own pre-registered sanity gate and thresholds
— not attempted here, and if pursued, the read must be pre-registered before
running it, exactly as Stage A itself was.

## Does everything degenerate together at extreme length?

Severely, but not to noise, for the variants measurable that far. All four dense
variants degrade 15–20× from ctx=256 to ctx=8192 (D0: 19.0×, D0p: 20.3×, D1: 20.3×,
Q0: 15.4×) — a model trained at ctx=256 clearly has little usable positional
structure left by ctx=8192, though it is not total collapse (ppl plateaus around
85–115, far below the ~4096 a uniformly random model would give). Per the Q0
section above, Q0's low ppl in this regime is a floor-control artifact of this
same degradation pattern, not a quality signal.

## The ctx=8192 ceiling — confirmed as architectural, not a bug

Unchanged from the original version; the review confirmed this section as
correct and disclosable as written. Extensive live debugging on Colab (memory
diagnostics inserted before/after every context length) ruled out
caching/fragmentation as the cause. The real explanation:
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
(`test_phase4_numpy.py`, the P4T1–T8 torch tests) before it's trustworthy. Given
the Q0 finding above, this would also only buy access to a rung outside the
interpretable range for quality claims — worth doing for the cost story and for
future longer-trained checkpoints, not for resolving the crossing question. Not
attempted here.

## Bottom line for the paper (revised)

1. **Raw ppl and the normalized metric now tell asymmetric-confidence stories, and
   neither survives as an unconditional claim.** Raw ppl: D1 leads S at every rung
   from 1024 through 4096, gap widening monotonically, and by ctx=4096 this is
   **seed-unanimous (3/3, one-tailed sign-test p=0.125 — not significant at
   conventional thresholds, but the most seed-consistent and paired result in
   this document) — see §2's carve-out below for how this survives the
   interpretable-range caution rather than being discounted by it.** Normalized
   metric: the pooled mean also shifts to D1's favor by ctx=4096, but **no rung
   from 512 through 4096 is seed-unanimous** — every one is a 2-of-3 split,
   carried mainly by one seed (1337) that favors D1 on this metric from ctx=512
   onward. Treat the raw-ppl trend as the stronger of the two and the
   normalized-metric trend as suggestive only. The claim that survives with full
   confidence is narrower than either original framing: *S degrades less than D1
   from its own baseline at ctx=512 through ctx=2048 on the pooled mean (peaking
   at 27% advantage at ctx=512), but this is not a seed-unanimous effect at any
   rung, and by ctx=4096 D1 leads on raw ppl with full seed consensus, read as a
   paired relative-rate-of-collapse result, not a capability claim (§2).*
2. **Ordering beyond ctx≈2048 is not reliably interpretable as extrapolation
   quality — but this governs cross-field ordering, not paired comparisons, and
   the two must not be conflated.** The no-interaction floor control Q0 rises
   monotonically in *pooled rank* with evaluation length, reaching 2nd of 6 by
   ctx=4096 and 1st of 4 at ctx=8192 — but the individual comparisons underneath
   that climb are themselves mean-driven, not seed-unanimous (Q0 beats S at
   ctx=2048 in 2/3 seeds; beats D1 at ctx=4096 in 2/3; beats D0p at ctx=4096 in
   only 1/3). **Explicit carve-out:** this caution applies to ranking the whole
   six-variant field against each other (reliable through ctx≤1024,
   reportable-with-caveat at ctx=2048, cost/collapse characterization only at
   ctx≥4096) — it does **not** apply to a paired two-variant sign test on
   identical windows (a third variant's rank moving doesn't undermine a pair's
   own comparison). Point 1's ctx=4096 raw-ppl finding is exactly such a pair,
   and stands *inside* the cost-only regime as a **relative-rate-of-collapse**
   statement, explicitly not a capability statement.
3. **The S-vs-X dissociation is intact through 4096 on both metrics but
   decaying** — not sharpened, contra this document's original wording — from
   60.3% at ctx=512 to 7.2% at ctx=4096, with one seed already crossing at 4096.
   This decay is in tension with the Stage A finding that X's out-of-distribution
   engagement is already collapsed by ctx=1024 (`RESULTS_phase4.md` §12.9):
   if engagement explained the behavioral gap, the gap should persist past 1024,
   not continue shrinking. Reported here as an open tension for the paper's
   mechanism section, not resolved.
4. **New cost-side finding, unchanged and confirmed:** K3's per-head score
   computation carries an `O(batch·T²·16)` memory ceiling against dense
   attention's `O(batch·T²)`, blocking evaluation past ctx≈4096 on a 40GB GPU —
   independent of and additive to the documented 2.39× wall-clock cost.
5. **A methodology concern was raised and killed, not just noted:** the per-ctx
   batch-size schedule at ctx=4096 was checked directly and confirmed to draw the
   same 256 windows for every variant regardless of batch configuration — the
   S-vs-D1 and S-vs-X comparisons at ctx=4096 are genuinely paired, not an
   artifact of differing batch sizes.
6. **Carried forward for Phase 5 metric pre-registration, not acted on here:**
   this document's own history is the argument for it. The original version
   treated the normalized metric as the safe fallback if raw ppl failed; it
   turned out to be the *less* seed-stable of the two, not the more robust one
   — it crosses in the same interval as raw ppl, and unlike raw ppl it is never
   seed-unanimous past ctx=256. Dividing by `e^val_loss` adds the variance of a
   second estimate rather than cancelling anything. Recommendation for whoever
   drafts Phase 5's metric pre-registration: **declare raw ppl (or raw loss) the
   primary metric and the normalized ratio a declared secondary, before any
   Phase 5 numbers exist** — not the reverse, and not decided implicitly by
   which one looks safer when a result is inconvenient.
7. **A third instance of a named propagation pattern, folded into
   `RESULTS_phase4.md` §11** rather than re-argued here: §9 limitation 5 ("seed
   variance in the dense family... sets a floor on readable margins for any
   dense comparison") was already on record before this document's S-vs-D1 and
   Q0-ranking work independently re-derived the same conclusion without citing
   it. The re-derivation was still the right thing to do; the citation should
   have been there too. §11 now names this as the third case and recommends a
   standing pre-write check: grep prior documents for a claim before
   re-deriving it cold.
