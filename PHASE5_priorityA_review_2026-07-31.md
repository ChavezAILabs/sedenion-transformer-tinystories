# Priority A review — chat-side analysis of `PHASE5_priorityA_crossing_2026-07-31.md`

**Author:** chat-side (Claude.ai), 2026-07-31. **Consumer:** Claude Code.
**Inputs read:** `PHASE5_priorityA_crossing_2026-07-31.md` (full),
`RESULTS_phase4.md` (full).
**Status of everything below:** analysis and recommendation only. Nothing here is
verified against the repo. Every proposed change goes through the validated chain
(`test_phase4_numpy.py`, P4T1–T8) before it is trusted, and every recomputation
below is arithmetic on the numbers *as printed in the Priority A table* — not a
re-read of `p4_artifacts/*/summary.json`. Reproduce from source before acting.

---

## 0. What holds up

Stated first so it is not lost in the criticism.

- **The protocol self-check is the right gate and it passed cleanly.** Reusing
  `phase4_grid.ce_loss` and the exact seed formula (`cfg.seed*1000+ectx`), then
  requiring ctx256/512/1024 to reproduce each checkpoint's own recorded
  `summary.json` numbers *to the digit* before trusting any new point, is the
  correct order of operations. It should be the standing pattern for any future
  eval-only run.
- **The ctx=8192 memory ceiling is a real, clean, disclosable finding.** The
  diagnosis (`prod` and `prod.pow(2)` both full-size and coexisting ⇒ peak ≈ 2×
  the `(B,H,T,T,16)` tensor, confirmed against the allocator's own "24.00 GiB")
  is specific, falsifiable, and independent of any ZDA claim. `O(batch·T²·16)`
  vs dense `O(batch·T²)` is a genuine cost-side result and pairs well with the
  already-documented 2.39× wall-clock overhead (`RESULTS_phase4.md` §4.4/§6.2).
- **Gate 0 was checked before running rather than assumed**, and the
  batch-size change was disclosed rather than buried. Both correct.
- **The core negative observation is correct as an observation:** D1 is ahead of
  S in raw ppl at 1024, 2048 and 4096, and the gap widens across those rungs.

---

## 1. BLOCKING — the normalized metric also crosses, between 2048 and 4096

The document's bottom line #1 concludes that the raw-ppl claim is dead and that
the claim "has to stay on the normalized metric (ppl@1024 / e^val_loss)."

**That metric was never computed at the new rungs. It does the same thing.**

Normalizing each variant by its own ctx=256 perplexity (pooled means, from the
document's own full results table):

| ctx | S norm | D1 norm | winner |
|---|---|---|---|
| 512 | 1.864 | 2.553 | S by 27.0% |
| 1024 | 5.721 | 6.509 | S by 12.1% |
| 2048 | 12.282 | 12.447 | S by 1.3% |
| 4096 | **19.205** | **17.040** | **D1 by 12.7%** |

Normalizing by `e^val_loss` instead (S: e^1.7980 = 6.0378; D1: e^1.6381 =
5.1452 — the metric exactly as `RESULTS_phase4.md` §6.1 defines it) gives the
same crossing in the same interval: S 12.409 vs D1 12.653 at 2048 (S ahead by
1.9%), S 19.403 vs D1 17.321 at 4096 (D1 ahead by 12.0%).

**Consequence.** The recommended fallback is not a fallback. The honest claim is:

> S's normalized extrapolation advantage over D1 is present at ctx=512 and
> ctx=1024, decays monotonically with context, and is gone by ctx=4096.

That is still statable and still worth publishing, but the **range travels with
the claim**, and the normalized metric can no longer be described as the robust
one. It is the one that survives *longer*, not the one that survives.

**Action:** add a normalized column to the Priority A results table at every rung
for every variant, recomputed from `p4_artifacts/*/summary.json` rather than from
the printed pooled means. Then rewrite bottom line #1.

---

## 2. BLOCKING — Q0 has already inverted the ordering by ctx=2048

The document records Q0's ctx=8192 result as "undirected, purely descriptive."
The inversion does not start at 8192. Ranking all variants at every rung
(1 = lowest ppl):

| ctx | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|
| Q0 rank, raw ppl | 6/6 | 6/6 | 5/6 | 4/6 | **2/6** | **1/4** |
| Q0 rank, normalized | — | 6/6 | 5/6 | **2/6** | **2/6** | **1/4** |

Full raw ordering for the record:

- ctx 1024: D0 < D0p < D1 < S < **Q0** < X
- ctx 2048: D0 < D0p < D1 < **Q0** < S < X
- ctx 4096: D0 < **Q0** < D0p < D1 < S < X
- ctx 8192: **Q0** < D0 < D1 < D0p

**Q0's rank is a monotone function of evaluation length on both metrics.** The
no-interaction floor control — the variant that is by construction the least
capable model in the grid, and last place at ctx=256 — climbs to first place.

Two readings, offered without adjudicating between them:

1. **Metric-artifact reading.** Past ~2048, ppl ordering ranks models by how much
   length-fragile structure they had to lose, not by capability.
2. **Real-property reading.** Q0 genuinely has better length robustness *because*
   it has no query-dependent interaction to break. A real model property — but
   one orthogonal to language-modeling quality.

**Both readings have the same consequence for the paper.** The headline
comparison — "at ctx=4096 all three seeds show D1 ahead of S" — is made in a
regime where the floor control is also ahead of D1. The observation is correct;
it cannot be read as evidence about extrapolation *quality*.

**Action:** declare an explicit **interpretable-range ceiling** for these
checkpoints and apply it consistently. Proposed: strict interpretability while Q0
is last (ctx ≤ 1024); reportable with an explicit caveat while Q0 is mid-pack
(ctx = 2048); cost-and-collapse characterization only beyond that (ctx ≥ 4096).
Add a "Q0 rank" row to any future long-context eval table as a standing canary.

This also weakens the S-vs-X section's "we cannot rule out a crossing beyond
4096" — a crossing out there would occur in a regime where the ordering has
already inverted, so it would not mean what the sentence implies.

---

## 3. ~~BLOCKING~~ RESOLVED 2026-07-31 by Claude Code — refuted by direct test

**Status update, added after the fact; original argument below left untouched.**
Tested directly per this section's own "Action" (below): a fixed-seed CPU
`torch.Generator` drawing 32×(batch=8) and a fresh same-seed generator drawing
128×(batch=2) produce **bit-identical** flattened index sequences — no GPU needed,
since the index draw in `ValOnlyDataset.get_batch` is CPU-side regardless of eval
device (the CUDA-Philox-block-alignment concern raised below does not apply here,
since it's never a CUDA generator in the first place). The ctx=4096 S-vs-dense
comparison is confirmed genuinely paired on identical windows, which is a
*stronger* result than an unblocked-but-unverified comparison would have been —
see `PHASE5_priorityA_crossing_2026-07-31.md`'s methodology note and
`phase4_priorityA_review_verify.py::check_3_batch_pairing` for the rerunnable
check. Left as BLOCKING below because that is what was actually claimed before
verification; do not read the original text as still live.

### 3 (original). the batch-size change lands on the only rung carrying the claim

Per the document's own methodology note: ctx ≤ 2048 ran the original protocol
(`batch_size=8, iters=32`) for everything. At ctx=4096, S/X ran `batch=2,
iters=128` while **dense variants ran the unmodified `batch=8`**.

So ctx=4096 is the **only rung where S/X and the dense family were evaluated
under different batch configurations** — and it is the rung supplying "all three
seeds show D1 ahead of S."

The document's defense (same N=256, each example's loss computed identically
regardless of grouping) is correct about *per-example* loss but presupposes the
**same 256 windows** are drawn. That presupposition is what makes this a paired
comparison at all: with eval seed `cfg.seed*1000+ectx`, S and D1 at a given run
seed and ectx receive the same generator seed, so at ctx ≤ 2048 they see
identical windows. At ctx=4096 they do not necessarily: 32 draws of 8 offsets and
128 draws of 2 offsets from the same generator are not guaranteed to yield the
same flat offset sequence, particularly on CUDA where the Philox counter advances
in fixed-size per-launch blocks rather than element-by-element.

If the window sets differ, the ctx=4096 S-vs-D1 comparison is **unpaired**,
against a between-seed S std of ±17.81 at that rung.

**Action — cheap and decisive.** Re-run D1 at ctx=4096 with `batch=2,
iters=128` and compare to the recorded `batch=8` figure.

- **Bit-identical** ⇒ concern is dead; record it as one disclosed line in the
  methodology note ("verified: offset draw is batch-size invariant at ctx=4096").
- **Not identical** ⇒ the dense ctx=4096 column must be recomputed at matched
  batch size before any S-vs-D1 claim at that rung stands.

No check needed at ctx ≤ 2048 (symmetric protocol). The same question applies at
ctx=8192, but that rung is outside the interpretable range per §2 anyway.

---

## 4. Gate 0 verifies the code path, not the behavior

Gate 0 reads: S, X and D1 "all use the same `R_8`/`omega_h` per-head
rotary-style relative encoding … confirmed by reading the code, not assumed."

Reading the code establishes that the same routine is called. It does not
establish that the routine behaves the same way, because X's generator is built
from the shuffled tensor.

This is in direct tension with a finding already on record in
`RESULTS_phase4.md` §12.5, which attributes part of the i.i.d.-null init skew to
S's **exact-norm-preserving** `R_8` rotation and notes that X's rotation is
documented by `phase4_X_positional_check.py` to **not** preserve norms or the
shared-phase null cleanly.

Not a contradiction in the code — identical code path, non-identical mathematics.
But Gate 0 as written reads as "S and X are positionally comparable," and the
repo's own diagnostic says they are not. This is the propagation failure mode
`RESULTS_phase4.md` §11 already names: a fact established correctly in one
document and absent from another.

**Action:** restate Gate 0 as *"same code path for all three variants; generator
properties differ between S and X — see `phase4_X_positional_check.py`"*, and
cross-reference §12.5.

**Secondary observation (speculation, labeled).** The new data argues *against*
the broken-positional-encoding account being the driver of the S-vs-X gap: a
generator that degrades with offset predicts the S-X gap **widens** with context.
It narrows — 60.3% → 48.0% → 23.5% → 7.2%. This is mildly exculpatory for X as a
control. The caveat is that the encoding damage may **saturate** within a few
positions (consistent with the "nullity destroyed within one position step" note
already on record), in which case the check is uninformative rather than
exculpatory. Do not carry this forward as evidence in either direction without a
direct measurement of how X's generator error scales with offset.

---

## 5. Reframe — the S-vs-X dissociation is intact but *decaying*, and that is in
tension with the Stage A mechanism reading

The document's bottom line #2 states the dissociation is "intact and, if
anything, sharpened." Intact, yes. Sharpened, no — the margin falls monotonically
and one seed already crosses at 4096.

The sharper issue is the interaction with `RESULTS_phase4.md` §12.9. Stage A
found X's out-of-distribution steering **AMBIGUOUS at 512 and COLLAPSED at 1024**
(z = −0.25, indistinguishable from no effect), and the pre-registered
mechanism-paper reading is "only S's engagement transfers out of distribution."

If that is the explanation for the behavioral gap, then beyond ctx=1024 — where X
has no engagement left to lose — the behavioral gap should persist or widen.
Observed instead:

| ctx | X worse than S by | X steering (Stage A) |
|---|---|---|
| 512 | 60.3% | AMBIGUOUS (z = −2.91) |
| 1024 | 48.0% | COLLAPSES (z = −0.25) |
| 2048 | 23.5% | not measured |
| 4096 | 7.2% | not measured |

**The behavioral gap is not tracking the steering gap.** It is largest at exactly
the two rungs Stage A measured, and decays to near-nothing at rungs Stage A did
not measure.

This is a **comparison across two independent measurements, not an attribution.**
It does not overturn Stage A's pre-registered call — that call was made under a
locked rule on locked rungs and stands as made. But it is an open tension that
belongs in the mechanism section of the paper, not omitted in favour of
"sharpened."

**Optional follow-up (inference only, no training):** run
`phase4_matched_N_trained.py --rungs 2048,4096` with the same
correlation-matched null, including the pre-registered init-mode sanity gate at
each new rung. If S's steering also decays toward collapse at 2048/4096, the
mechanism story becomes internally consistent (engagement and behavior decay
together). If S's steering holds at 4096 while the behavioral gap has closed to
7%, the mechanism story has a real problem. **Pre-register the read before
running it** — this is a rung extension of an already-locked instrument, and it
should not become a post-hoc search for a rung where the story works.

---

## 6. Methodology notes (non-blocking)

1. **Pool in log space.** The results table averages perplexities across seeds.
   Perplexity is exponential in loss, so the arithmetic mean is worst-seed
   dominated — at S/4096 (±17.81) this is not negligible. Report
   `exp(mean(ln ppl))` (equivalently, mean cross-entropy) and give the std in
   loss space, where it is the natural scale.
2. **Lead with the paired per-seed statement.** The document already does the
   right thing at ctx=4096 ("all three seeds show D1 ahead"). With n=3, the
   3/3 sign statement on a paired comparison is stronger and more honest than
   mean ± std, and it should be the primary presentation at every rung — subject
   to §3 confirming the pairing actually holds at 4096.
3. **Effective sample size is constant in windows, not tokens.** N=256 windows at
   every rung means 256 independent samples at every rung, regardless of the 16×
   token growth from 256 to 4096. Do not read the growing std as an artifact of
   thinner evaluation; it is not, but neither is it cleanly attributable to model
   divergence. Leave it descriptive.
4. **Chunking `scores()` is the correct follow-up and correctly not attempted
   here.** Tiling over the sequence dimension inside `K3Attention.scores()`
   touches the validated core kernel and must go back through the
   NumPy-mirror-first chain. Given §2, note that it would buy access to a rung
   that is outside the interpretable range — worth doing for the cost story and
   for future longer-trained checkpoints, not for resolving the crossing
   question.

---

## 7. Recommended action list, in order

| # | Action | Cost | Blocks |
|---|---|---|---|
| 1 | Matched-batch check: D1 at ctx=4096, `batch=2, iters=128`, test bit-identity vs recorded | minutes, GPU | §3 — the headline claim |
| 2 | Recompute normalized metric at all rungs, all variants, from `summary.json` | minutes, CPU | §1 — bottom line #1 |
| 3 | Add Q0-rank-by-rung row; adopt an explicit interpretable-range ceiling | minutes | §2 — every long-context claim |
| 4 | Restate Gate 0 as code-path-only; cross-ref `phase4_X_positional_check.py` and §12.5 | minutes | §4 |
| 5 | Rewrite bottom lines #1 and #2 per §1 and §5 | — | paper framing |
| 6 | Log-space pooling for all pooled ppl figures | minutes | §6.1 |
| 7 | *Optional, pre-register first:* Stage A rung extension to 2048/4096 | inference only | §5 tension |

---

## 8. Proposed replacement bottom lines

Offered as drafts for Claude Code to check against the repo, not as agreed text.

1. **The absolute-ppl framing does not survive, and neither does the normalized
   fallback past ctx=2048.** D1 leads S in raw ppl at every rung from 1024
   through 4096 with a widening gap; on the normalized metric S leads at 512 and
   1024, ties at 2048, and trails at 4096. The claim that survives is
   range-bounded: *S degrades less than D1 from its own baseline at ctx=512 and
   ctx=1024, and that advantage decays to zero by ctx=4096.*
2. **Ordering beyond ctx≈2048 is not interpretable as extrapolation quality.**
   The no-interaction floor control Q0 rises monotonically in rank with
   evaluation length on both metrics, reaching second place at 4096 and first at
   8192. Long-context ordering is dominated by how much length-fragile structure
   a model had to lose. Claims about extrapolation quality are confined to
   ctx ≤ 1024, with 2048 reportable under caveat.
3. **The S-vs-X dissociation is intact through 4096 on both metrics but
   decaying**, from 60.3% at 512 to 7.2% at 4096, with one seed crossing at 4096.
   This decay is in tension with the Stage A reading that only S's engagement
   transfers out of distribution, since X's engagement has already collapsed by
   1024 while the behavioral gap continues to close. Reported as an open tension.
4. **New cost-side finding:** K3's per-head score computation carries an
   `O(batch·T²·16)` memory ceiling against dense attention's `O(batch·T²)`,
   blocking evaluation past ctx≈4096 on a 40GB GPU — independent of and additive
   to the documented 2.39× wall-clock cost. Unchanged from the original document;
   this one stands as written.
