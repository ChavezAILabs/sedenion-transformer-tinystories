# RESULTS — Phase 4 (ZDA grid, 18/18)

**Status:** Verified against the repo by Claude Code, 2026-07-27 (see §12).
Originally drafted chat-side; every number was independently reproduced from
`p4_artifacts/*` by re-running `phase4_grid.py`'s own `grand_summary`/
`h4c_readout` functions, or recomputed from source (`sedenion_kernel.py`,
`phase4_layers.py`) rather than accepted on the draft's word. Two real errors
found and corrected in place (§6.1, §8.1); everything else — including every
number in §3–§5, §7, and §8.3 — reproduced exactly. §8.2's qualitative claims
hold but its point-estimates are Monte-Carlo figures with sampling-method
sensitivity; see the footnote there.

**Grid:** complete, 18/18. Gate lifted; grand summary read out.
**Script provenance:** all 18 runs produced by `phase4_grid.py` at
sha256 `65d419e05b6ec74d…`. Readout corrected at 415506e (H4c seed-discovery
bug, §7.1). No training was re-run after the fix.
**Convention:** Baez throughout.

---

## 1. Executive summary

Five findings.

1. **The pre-registered hypothesis H4a is negative, decisively.** S trails the
   flops-matched dense baseline by 0.1600 in validation loss against a
   pre-registered margin of 0.0076 — a factor of 21. H4b′ never evaluated.
2. **The mechanism nonetheless engages.** H4c passes: six attention heads in S
   reliably descend toward the zero-divisor structure across all three seeds,
   five of them in layer 0.
3. **The mechanism does not pay.** Every dense baseline beats both tensor
   variants on validation loss, and D0 does so at 13.58% fewer flops per token.
4. **The primary positive is a controlled double dissociation between S and its
   own shuffled control X**, 3/3 seeds with complete separation on all three
   measured axes. This is the only comparison in the grid in which a single
   variable differs.
5. **Cost is real and exceeds the pre-registered bound:** S runs at 2.39×
   D0p wall-clock, against a 2× ceiling, at identical nominal flops/token to D1.

Net: *the architecture does the thing it was built to do, and doing it makes
the language model worse.*

---

## 2. Setup

Six variants × three seeds (1337, 1338, 1339). TinyStories, 4k byte-level BPE,
context 256, 18,311 steps × 16,384 tokens = 300.0M tokens per run, eval every
1,526 steps. Length generalization measured at 512 and 1024. TF32 disabled
(spec §9.2).

| variant | role | d_model | mlp | params | flops/tok |
|---|---|---|---|---|---|
| D0p | dense, primary baseline | 384 | 1536 | 13,784,064 | 26,738,688 |
| D0 | dense, reference only (not graded) | 384 | 1536 | 13,784,064 | 26,738,688 |
| D1 | dense, flops-matched to S | 384 | 1992 | 15,888,048 | 30,941,184 |
| **S** | true sedenion structure tensor | 384 | 1824 | 13,785,828 | 30,941,184 |
| **X** | shuffled-tensor control | 384 | 1824 | 13,785,828 | 30,941,184 |
| Q0 | no-interaction floor (not graded) | 384 | 1727 | 13,782,906 | 25,554,432 |

S and X are identical in every dimension the harness controls — d_model, mlp
width, parameter count to the digit, flops/token, and wall-clock to within 6
seconds. **The structure tensor is the only difference between them.**

---

## 3. Grand summary

| variant | n | val loss (mean ± std) | ppl@512 | ppl@1024 | k1 min | wall h |
|---|---|---|---|---|---|---|
| D0p | 3 | 1.6620 ± 0.0066 | 9.84 | 28.90 | 2.54e+00 | 0.60 |
| D0 | 3 | 1.6400 ± 0.0065 | 7.89 | 20.69 | 2.54e+00 | 0.60 |
| D1 | 3 | 1.6381 ± 0.0042 | 13.34 | 34.04 | 2.53e+00 | 0.67 |
| **S** | 3 | 1.7980 ± 0.0033 | 11.37 | 34.90 | 1.33e+00 | 1.44 |
| **X** | 3 | 1.7632 ± 0.0012 | 18.22 | 51.64 | 2.42e+00 | 1.44 |
| Q0 | 3 | 1.8211 ± 0.0079 | 20.54 | 45.88 | 0.00e+00 | 0.54 |

No run diverged. D0 is reference only and Q0 is the no-interaction floor;
neither is graded.

---

## 4. Pre-registered verdicts (spec §5 v1.0, frozen)

### 4.1 H4a — NEGATIVE

- **H4a-1:** S − D1 val = **+0.1600**, margin 0.0076 → negative
- **H4a-2:** S − D0p ppl@1024 = **+6.001** (margin 10.216) with |S − D0p| val
  = 0.1360 (within the 0.0104 parity condition) → negative

Clause 1 fails by a factor of 21, not marginally. The pre-registration
functioned as intended.

### 4.2 H4b′ — NOT EVALUATED

Conditional on an H4a win. Correctly skipped.

### 4.3 H4c — PASS

Frozen rule: pass iff ≥1 (layer, head) whose end-of-training p5(r²) is below
its step-0 p5(r²) by more than 2× the pooled-across-seeds std of that head's
step-0 p5, **in all three seeds, same head**.

Six heads cross:

| head | step-0 p5 | final p5 | drop | margin | drop / margin |
|---|---|---|---|---|---|
| L0H1 | 0.7002 | 0.3957 | 0.3045 | 0.0167 | 18× |
| L0H2 | 0.6923 | 0.4687 | 0.2236 | 0.0062 | 36× |
| L0H3 | 0.7016 | 0.5631 | 0.1385 | 0.0215 | 6× |
| L0H4 | 0.6962 | 0.4287 | 0.2675 | 0.0416 | 6× |
| L0H5 | 0.7130 | 0.4100 | 0.3030 | 0.0097 | 31× |
| L2H3 | 0.6969 | 0.5349 | 0.1620 | 0.0247 | 7× |

**Five of six are layer 0** — the layer A3-revised targets.

The conjunction across seeds is doing real filtering rather than
rubber-stamping: L1H5 drops 0.0683 against a margin of 0.0055 (12× the bar on
the mean) and still fails, because it did not cross in every seed.

### 4.4 Wall-clock bound — EXCEEDED

S / D0p = **2.39×**, against a 2× ceiling. At *identical nominal flops/token*
to D1 (30,941,184), S takes 1.44 h against D1's 0.67 h — a 2.16× penalty
attributable to hardware utilization of the tensor product, not to arithmetic
volume. See §6.2.

---

## 5. Primary positive: the S–X double dissociation

X fits the training distribution better; S extrapolates better. 3/3 seeds,
**complete separation on all three axes**.

| measure | S (1337 / 1338 / 1339) | X (1337 / 1338 / 1339) | winner |
|---|---|---|---|
| val loss | 1.7979 / 1.8014 / 1.7949 | **1.7638 / 1.7640 / 1.7619** | X |
| ppl@512 | **12.137 / 11.216 / 10.75** | 18.665 / 18.045 / 17.959 | S |
| ppl@1024 | **39.806 / 34.376 / 30.52** | 52.219 / 51.605 / 51.096 | S |

X's worst val (1.7640) beats S's best (1.7949). S's worst ppl@512 (12.137)
beats X's best (17.959); same at 1024 (39.806 vs 51.096). Nine runs, three
axes, zero crossings.

### 5.1 Why the crossing matters

Any single-factor account — capacity, conditioning, optimization difficulty,
task difficulty — predicts a **uniform** ordering. A crossing rules that class
out. In particular it defeats the strongest available deflationary explanation:
X is non-unital, non-flexible, non-power-associative, and ~16× worse
conditioned at the median (§8.2), but a capacity handicap is monotone and
cannot produce *better* in-distribution and *worse* out of it.

What remains is that the two tensors induce genuinely different **inductive
biases** — they trade in-distribution fit against length extrapolation in
opposite directions. That is a claim about what kind of model results, not how
good a model results.

### 5.2 Asymmetry

The trade is lopsided. X gains ~2% in-distribution and loses 48–60% on
extrapolation. Worth stating plainly rather than reporting the dissociation as
symmetric.

### 5.3 Variance asymmetry (descriptive)

S's extrapolation is far noisier than X's: at 1024, S spans 30.52–39.806 (30%
of its minimum) while X spans 51.096–52.219 (2%). S is variable and better; X
is stable and bad. **Recorded as an observation, not an interpretation** —
n = 3 will not support a mechanism reading, and better models may simply have
more room to differ.

---

## 6. Secondary findings

### 6.1 MLP width badly hurts length generalization — corrected

D0 (mlp 1536) beats D1 (mlp 1992) at both rungs — 7.89/20.69 against
13.34/34.04 — with **fewer parameters and fewer flops**.

**Correction (Claude Code, 2026-07-27): the draft's claim that rung
performance "sorts monotonically by MLP width across D0 (1536) → S (1824) →
D1 (1992) on both rungs" is false at ppl@1024.** It holds at ppl@512
(7.89 < 11.37 < 13.34, exactly D0 < S < D1). At ppl@1024 the true order is
**D0 (20.69) < D1 (34.04) < S (34.90)** — S has the *worst* ppl@1024 of the
three despite a narrower MLP than D1 (1824 vs 1992). Recomputed directly from
`p4_artifacts/{D0,D1,S}_seed*/summary.json`.

This changes the interpretation, not just the arithmetic: a pure width
confound predicts D0 < S < D1 at both rungs. It holds at 512 and fails at
1024, where S — the narrower of the two — is worst. That is evidence
*against* MLP width alone driving S's ppl@1024 behavior, not for it. Q0 (mlp
1727, worst rungs of any variant) already showed the monotonicity doesn't
hold generally; this sharpens that into a specific, checkable failure at the
1024 rung. Width remains a live confound worth isolating (Phase 5's D1824,
§9.1) — this correction says the grid data don't yet support the strong
"drives the outcome" reading the draft gave it.

**Moot for H4a, which fails on clause 1 regardless.**

### 6.2 Wall-clock cost at nominal flops parity

S and X take 1.44 h against D1's 0.67 h at identical flops/token. The tensor
product is poorly hardware-utilized; nominal FLOP accounting understates its
cost by ~2.2×. Any efficiency claim for this architecture must be made in wall
time, not flops.

### 6.3 Bit-exact reproducibility across sessions and hardware

Seed 1339's D0p run was re-executed from scratch in a fresh Colab session, on
a different physical A100, with the data cache re-copied from Drive. All
eleven eval points reproduced **identically** — train loss, val loss, and k1,
every field, every step, to four decimals. Determinism holds across runtime,
hardware instance, and data transport. Not free at this scale; worth a line.

---

## 7. Corrections and audit trail

Negative results and corrections receive the same care as positive ones
(standing order §7).

### 7.1 The H4c readout bug

The first 18/18 readout printed H4c as "negative" with every per-head margin
`nan` and numpy "degrees of freedom ≤ 0" warnings. Root cause: `h4c_readout()`
looped over the `seeds` parameter, which flows from `main()`'s `--seeds` CLI
argument — the same argument controlling the training loop. The invocation that
completed the grid passed `--seeds 1339`, collapsing the grading seed axis to
length 1; `std(ddof=1)` on N=1 divides by zero. Since `nan` is never `<`
anything, every head's comparison fell through to `False` and "negative"
printed **without a single head having been tested**.

H4a and H4b′ were unaffected: they aggregate via `glob.glob` over
`summary.json` and auto-discover every run.

Fixed at 415506e: seed discovery from disk, refusal to grade unless exactly 3
seeds are found, and a guard that raises rather than prints on any non-finite
margin. The statistic, margin, and threshold are untouched. **H4c's verdict
changed from an unadjudicated "negative" to a genuine PASS.**

Corrected readout confirmed H4a bit-identical (+0.1600, margin 0.0076) and the
grand summary table unchanged to every digit.

### 7.2 Amendment provenance

A1, A2, A3, and A3-revised are labeled **post-hoc, not blind**. Partial
seed-1339 data (D0p, 16,750 of 18,311 steps) existed before the amendments were
drafted. Whether that partial output was consulted is unknown; per §5's rule,
uncertainty resolves to the weaker label. Recorded at 415506e's predecessor.

### 7.3 Descent range correction

S's r²_min descent across three seeds is **77.7×–121.3×**, not the 99–117×
recorded in earlier amendments. Seed 1339's 77.7× falls below the previously
recorded range. Correct wherever cited.

---

## 8. Mechanism analysis and its limits

### 8.1 What H4c does and does not show

H4c tests **reliable** movement, not **large** movement: the bar scales to each
head's own step-0 variance. X crosses the same bar in two heads (L1H1, L2H0) in
its descriptive trace. So the criterion alone does not discriminate S from X.

The discrimination is entirely in **magnitude**, which H4c does not test:

- S per-head drops: 0.1385 – 0.3045
- X per-head drops: 0.0171 – 0.0505

S's *smallest* drop is 2.7× X's *largest*; median ratio ≈ 7×.

**Correction (Claude Code, 2026-07-27): the table below is layer-0 only, not
whole-model.** Recomputing r²_min pooled across all 6 layers gives different
@0 values (e.g. S 1337: 0.1116, not 0.1529) — the numbers as originally
labeled are exactly reproduced only by taking `diag[0]` (layer 0) at each
step, not `min` over all layers' diag entries. This matches the independent
finding already on record in `PHASE5_stage0_findings_2026-07-26.md` §1
(same fact, established there for a different purpose). The figures
themselves are unchanged and correct; only the "whole-model" label was wrong.

Layer-0 r²_min traces:

| run | r²_min @0 | @end | descent | min ever | max frac < 1e-2 |
|---|---|---|---|---|---|
| S 1337 | 0.1529 | 0.00126 | 121.3× | 0.00093 | 0.0033 |
| S 1338 | 0.1382 | 0.00140 | 98.5× | 0.00140 | 0.0026 |
| S 1339 | 0.1439 | 0.00185 | 77.7× | 0.00122 | 0.0026 |
| X 1337 | 0.0522 | 0.05022 | 1.0× | 0.03302 | 0.0000 |
| X 1338 | 0.0820 | 0.03751 | 2.2× | 0.03311 | 0.0000 |
| X 1339 | 0.0470 | 0.04655 | 1.0× | 0.02818 | 0.0000 |

X never places a pair below 1e-2 in any seed — its best-ever value is 2.8×
above the threshold. X's descent is 1.0×, 2.2×, 1.0× and **non-monotonic**
(min-ever well below endpoint), i.e. a random walk rather than a descent that
stalls at a barrier.

### 8.2 What X actually is

X is not a relabeled sedenion. Measured properties, both readings of the
shuffle spec:

| property | S | X |
|---|---|---|
| dimension | 16 | 16 |
| nonzero entries | 256 | 256 |
| ‖C‖_F | 16.000 | 16.000 |
| E‖x⊛y‖, random unit x,y | 0.996 | 0.986–0.988 |
| dim of zero-divisor variety | 14 | 14 |
| two-sided identity | yes | **no** |
| flexible, (xy)x = x(yx) | exact (4e-16) | **violated, ~1.5** |
| power-associative | exact (4e-16) | **violated, ~1.8** |
| median cond(L_x) | 2.8 | **44–50** |

X *does* possess a zero-divisor variety (infimum ≈ 0, confirmed by two
independent implementations), of the same dimension as S's. But **dimension 14
is generic**: 30 ambient − 16 equations, and a dense Gaussian bilinear map has
it too. Possessing a ZD variety distinguishes nothing and no claim should rest
on it.

**Implication for the X-inequivalence certificate:** dimension is not a
discriminating invariant. Any certificate must rest on isometry type and
homogeneity (Reggiani's G₂ result), never on a dimension count, which passes
for random noise.

**Verification footnote (Claude Code, 2026-07-27):** every qualitative claim
in the table above is confirmed by independent recomputation from
`structure_tensor()`/`shuffled_structure_tensor(seed)` — X has a left but not
a two-sided identity, is not flexible, is not power-associative, and is
markedly worse-conditioned than S (~15× at the median, same order of
magnitude as the draft's "44–50" vs the repo's actual 40.6–43.2 across the
three grid seeds; S itself reproduces at 2.68, not 2.8). The exact
point-figures for E‖x⊛y‖, median cond(L_x), and the flexible/power-associative
residuals are Monte-Carlo estimates (random unit-vector sampling); an
independent draw reproduces the same order of magnitude and the same
qualitative verdicts but not the same digits, since the draft doesn't specify
its sample count or RNG. Treat this row as **qualitatively confirmed, point
estimates approximate** — unlike §3–§5, §7, and §8.3, which reproduce exactly
because they come from deterministic training logs or exhaustive enumeration.

### 8.3 Convention reconciliation (closes the 84-vs-336 flag)

Enumerated directly in Baez over elements e_a ± e_b, 1 ≤ a < b ≤ 15:

| convention | count |
|---|---|
| ordered pairs (x,y) with x⊛y = 0 | **336** ← repo |
| unordered {x,y} | 168 |
| distinct zero-divisor elements x | **84** ← literature |
| distinct index sets {a,b} | 42 |

336 = 84 × 2 (ordering) × 2 (sign on the second factor). Both counts are
correct for what they count; label rather than reconcile. Literature exemplar
pairs (e₃+e₁₀)(e₆−e₁₅) and (e₅+e₁₀)(e₆+e₉) verified to annihilate exactly
under Baez — no conversion was required for them.

**Verified against the repo's own enumerator (Claude Code, 2026-07-27):** an
independent vectorized enumeration over all 210 candidate elements e_a ± e_b
(1 ≤ a < b ≤ 15) reproduces 336 ordered pairs, 168 unordered pairs, 84
distinct zero-divisor elements, and 42 distinct index sets exactly, with zero
self-paired instances (no x with x⊛x = 0). `sedenion_kernel.py`'s own
exhaustive search independently agrees (336, same worked example
`(e1+e10)(e4-e15)=0`). Both exemplar pairs above also reproduce exactly
(max|product| = 0.0). *Restricted to two-term elements, the form the
literature counts.*

---

## 9. Limitations

1. **No valid-algebra-wrong-frame control (the V3 analogue).** X differs from S
   in many ways simultaneously (§8.2). The dissociation attributes the trade to
   *the structure tensor*, **not** to zero-divisor structure specifically. This
   is the single largest limit on what Phase 4 can claim.
2. **n = 3.** Complete separation across three seeds per cell is strong for this
   scale but is not a significance test.
3. **Both tensor variants are worse language models than every dense baseline.**
   The dissociation is between S and its own control, not against a competitive
   baseline.
4. **Can't vs doesn't is unresolved.** Whether X *can* reach low r² under the
   real parameterization was never measured. Two seeds converging near 0.033
   suggested a floor; seed 1339's 0.02818 broke that reading. Open.
5. **Seed variance in the dense family is large** — D0p ppl@512 swings 46%
   between seeds — which sets a floor on readable margins for any dense
   comparison.
6. **A1/A2/A3 are post-hoc**, not blind (§7.2).

---

## 10. Open items

| item | cost | blocks | status |
|---|---|---|---|
| Constrained-infimum check (can't vs doesn't) | inference only | §8.1 framing | **partially done** — global variant (fresh-init, many restarts) unblocked per `SESSION_CONTINUATION_HANDOFF_2026-07-26.md` §3.1, not yet run; local variant (from trained checkpoints) still blocked, no checkpoints synced locally |
| cond(L_x) at the real init distribution | inference only | §8.2 attribution | **done** 2026-07-26 — confirmed real, not a sampling artifact (`PHASE5_stage0_findings_2026-07-26.md` §2); ~15× gap at real init matches the free-sphere prediction to ~1% |
| ~~Verify §8.3 against repo enumerator~~ | minutes | prior-art §8.4.8 | **done** 2026-07-27, this pass — see §8.3 |
| A3-revised γ dose-response | inference only, plausibly free tier | — | open |
| V3-analogue control | new experiment | §9.1 — Phase 5 scope | open — reopened as V3b, see `SESSION_CONTINUATION_HANDOFF_2026-07-26.md` §3.2 |

---

## 11. Methodology notes worth carrying forward

- Pre-registering diagnostics on an unobserved mechanism produced a criterion
  (H4c) that tests reliability rather than magnitude, and therefore does not
  discriminate the variant it was designed to characterize from its control.
  The magnitude statistic had to be added post-hoc (A1).
- A verdict was emitted from a `nan` statistic (§7.1). Any grading path should
  raise on non-finite values rather than fall through comparison semantics.
- A parameter shared between an execution loop and a readout loop is a latent
  correctness hazard: narrowing the run silently narrowed the grading.

---

## 12. Verification log (Claude Code, 2026-07-27)

Method: reproduce, don't proofread. Every number with a deterministic source
was regenerated from that source rather than checked by eye.

- **§2 setup table** (params/flops/d_model/mlp per variant) — reproduced
  exactly via `phase4_grid.py --match_table`.
- **§3 grand summary, §4.1 H4a, §4.3 H4c, §4.4 wall-clock** — reproduced
  exactly by importing `phase4_grid` and calling its own `grand_summary()`
  and `h4c_readout()` against `p4_artifacts/` directly (no training,
  no GPU). Every digit in every table matched, including all six H4c
  crossing heads, their step-0/final p5 values, and the L1H5 near-miss.
- **§5 double-dissociation per-seed table** — reproduced exactly from each
  run's `summary.json` (`final_val_loss`, `length_gen.ctx512.ppl`,
  `length_gen.ctx1024.ppl`).
- **§6.1** — **error found and corrected**: the "sorts monotonically ...
  on both rungs" claim is false at ppl@1024 (true order D0 < D1 < S, not
  D0 < S < D1). Verified against the same three variants' `summary.json`
  files. Correction changes the interpretation (evidence against a pure
  width confound at that rung, not for one).
- **§7.1–§7.3** — cross-checked against `phase4_grid.py`'s own in-code
  postmortem comment on the H4c bug (verbatim match) and against the
  layer-0 r²_min recomputation below; the 77.7×–121.3× range reproduces
  exactly.
- **§8.1** — **label error found and corrected**: table is layer-0 r²_min,
  not whole-model (pooled-across-layers r²_min gives different §0 values).
  The figures themselves were already correct; only "whole-model" was wrong.
  Same fact independently on record in `PHASE5_stage0_findings_2026-07-26.md`
  §1. Recomputed from `eval_log.jsonl`'s `diag[0].r2_min` at first/last/every
  step per run.
- **§8.2** — qualitative claims (non-unital, non-flexible, non-power-
  associative, ~15× worse conditioned) confirmed by independent
  recomputation from `structure_tensor()`/`shuffled_structure_tensor(seed)`
  for all three grid seeds. Point estimates are Monte-Carlo and don't
  reproduce to the digit (see footnote in §8.2) — flagged, not corrected,
  since the draft doesn't specify a sample count to reproduce exactly.
- **§8.3** — reproduced exactly: independent 210-element vectorized
  enumeration gives 336/168/84/42 with 0 self-paired instances, agreeing
  with `sedenion_kernel.py`'s own exhaustive search and with both literature
  exemplar pairs annihilating exactly.
- **§6.3** (bit-exact cross-hardware reproduction) and **§7.2** (amendment
  post-hoc labeling) were not independently re-verified this pass — both
  describe Colab-side events already recorded as confirmed in
  `SESSION_CONTINUATION_HANDOFF_2026-07-26.md` §2.1/§2.3 and not
  re-checkable from local artifacts alone.

**Net: two corrections (§6.1, §8.1), one caveat (§8.2), everything else in
§3–§5, §7, and §8.3 exact.** The document's headline claims (H4a negative,
H4c pass, the S–X double dissociation, wall-clock exceeding its bound) are
all unaffected by the corrections.
