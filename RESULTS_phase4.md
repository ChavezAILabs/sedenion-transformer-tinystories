# RESULTS — Phase 4 (ZDA grid, 18/18)

**Status:** Verified against the repo by Claude Code, 2026-07-27, four passes
(see §12–§12.3). Originally drafted chat-side; every number was independently
reproduced from `p4_artifacts/*` by re-running `phase4_grid.py`'s own
`grand_summary`/`h4c_readout` functions, or recomputed from source
(`sedenion_kernel.py`, `phase4_layers.py`) rather than accepted on the
draft's word. Two real errors found and corrected in place (§6.1, §8.1;
§6.1 narrowed further on chat-side review, §12.1); everything else —
including every number in §3–§5, §7, and §8.3 — reproduced exactly.
§8.2's `cond(L_x)` is now a **proved closed form** (BCDI 2009, §12.1), not a
Monte-Carlo estimate, and gained a second exact companion result — the
annihilator dimension (S: 4, X: 1, both confirmed by direct measurement,
§12.3). §8.2's other two point-figures (E‖x⊛y‖, flexible/power-assoc
residuals) remain Monte-Carlo with sampling-method sensitivity.

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

**Correction (Claude Code, 2026-07-27, narrowed 2026-07-27 per chat-side
review): the draft's claim that rung performance "sorts monotonically by MLP
width across D0 (1536) → S (1824) → D1 (1992) on both rungs" is false at
ppl@1024, in raw ppl@1024.** It holds at ppl@512 (7.89 < 11.37 < 13.34,
exactly D0 < S < D1). At ppl@1024 the true mean order is **D0 (20.69) <
D1 (34.04) < S (34.90)**. Recomputed directly from
`p4_artifacts/{D0,D1,S}_seed*/summary.json`.

**Two qualifications, both required for the claim to be precise:**
1. **Metric.** This is raw ppl@1024, the length-generalization rung reported
   in §3's grand summary — not the normalized-extrapolation metric of §1's
   executive summary (ppl@1024 / e^val, i.e. degradation from each variant's
   own in-distribution fit). The two **disagree on this exact pair**:
   normalized extrapolation puts S ahead of D1 (5.78× < 6.62×, S degrades
   *less* from its own worse baseline), while raw ppl@1024 puts D1 ahead of S
   (34.04 < 34.90, D1 is the lower absolute number). Both are correct; they
   answer different questions. Any width account has to explain both
   orderings, not just one.
2. **Spread.** The D1-vs-S gap at ppl@1024 is ~2.5% of the mean, and the
   grand summary only carries a reported std for val loss, not for
   perplexity — per-seed ppl@1024 is highly variable for both (S: 30.52,
   34.376, 39.806; D1: 24.628, 35.061, 42.442 — see §5.3's variance-asymmetry
   note). **The robust claim is that monotonicity by width fails at
   ppl@1024. "S is the worst of the three" is not robust without a reported
   spread and should not be read as more than the raw mean ordering.**

This still changes the interpretation, not just the arithmetic: a pure width
confound predicts a single consistent D0 < S < D1 ordering. Raw ppl@1024
breaks that ordering (D1 < S in the mean), and the normalized metric breaks
it in the *other* direction (S < D1). That two-way disagreement — not the
weaker "S is worst" reading — is the evidence against a simple width account,
and it is what Phase 5's width-isolation run (D1824, §9.1) needs to be
designed against. Q0 (mlp 1727, worst rungs of any variant) already showed
the monotonicity doesn't hold generally; this sharpens that into a specific,
checkable, metric-dependent failure at the 1024 rung.

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
independent implementations), of the same **total** dimension as S's. **The
total's genericity was the wrong reason for dismissing it, though the
dismissal itself survives (refined 2026-07-27, §12.3):** 30 ambient − 16
equations gives 14 for any bilinear map, S included — but for S that 14
decomposes as 11+3 (an 11-dimensional base, forced three independent ways —
Reggiani's G₂ isometry, Koebisu's V₂(ℝ⁷) projection, BDI's exact
multiple-of-4 annihilator bound — plus a 3-dimensional fiber), while a
generic bilinear map decomposes as 14+0 (full-dimensional base, no fiber).
Same total, opposite shape. Possessing a ZD variety of dimension 14 still
distinguishes nothing on its own; the *shape* of that 14 does, and now has a
direct numeric handle — see §12.3.

**Implication for the X-inequivalence certificate:** total dimension is not
a discriminating invariant. Any certificate must rest on isometry type and
homogeneity (Reggiani's G₂ result) or on the shape of the decomposition
(§12.3's annihilator-dimension result), never on a bare dimension count,
which passes for random noise.

**Verification footnote (Claude Code, 2026-07-27):** every qualitative claim
in the table above is confirmed by independent recomputation from
`structure_tensor()`/`shuffled_structure_tensor(seed)` — X has a left but not
a two-sided identity, is not flexible, is not power-associative, and is
markedly worse-conditioned than S (same order of magnitude as the draft's
"44–50" vs the repo's actual 40.6–43.2 across the three grid seeds).

**Update, same day, later pass: `cond(L_x)` for S is no longer a sampled
estimate — it is closed-form, via BCDI 2009 (arXiv:0905.2987) Cor. 7.3/Prop.
3.10.** For any nonzero sedenion `v = (a₀, u, b₀, w)` (CD-doubled into two
octonion halves, `u = v[1:8]`, `w = v[9:16]` their imaginary parts), left
multiplication `L_v` has **exactly three eigenvalues of `L_vᵀL_v`, at fixed
multiplicities (8, 4, 4) for every nonzero v**: `1` (×8), `1+S(v)` (×4),
`1−S(v)` (×4), where `S(v) = 2·√(‖u‖²‖w‖² − ⟨u,w⟩²) / ‖v‖²`. Hence
`cond(L_v) = √((1+S)/(1−S))` exactly. **Verified against this repo's actual
`structure_tensor()` to floating-point precision**: max eigenvalue error
3.6e-15 and max condition-number error 9.4e-12 over 5000 random unit v (the
theorem's own precision claim, not approximate agreement). The eigenvalue
multiplicity signature (8,4,4) itself is theorem-fixed — confirmed at
**every one of 200 sampled points**, no exceptions — while the shuffled
tensor X shows **16 distinct eigenvalues (no degeneracy at all) at every one
of 200 sampled points, all three grid seeds** — a clean, non-overlapping
binary discriminator, sharper than dimension or the earlier accessibility
statistics. Recomputing the median with this closed form (N=5000,
exact per-point) gives **2.7758**, superseding the earlier N=500 estimate
(2.68) and close to the draft's 2.8; the remaining sampling dependence is in
which random directions are drawn; there is no known closed form for the
*median itself* over the uniform sphere, only for `cond(L_v)` at each v.

The other two point-figures (E‖x⊛y‖, and the flexible/power-associative
residuals) remain Monte-Carlo estimates without a closed form found so far;
an independent draw reproduces the same order of magnitude and the same
qualitative verdicts but not the same digits. Treat those two rows as
**qualitatively confirmed, point estimates approximate**; `cond(L_x)` is now
**exact per-point, theorem-backed**, unlike §3–§5, §7, and §8.3's
deterministic-log/enumeration exactness, but exact in its own, stronger
sense (a proved closed form, not just a reproduced sample).

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
- **The §8.1 label error's failure mode was propagation, not discovery**
  (chat-side observation, 2026-07-27): the layer-0-vs-whole-model fact was
  already correctly established in `PHASE5_stage0_findings_2026-07-26.md`
  §1 before this document was drafted. It just hadn't been carried into
  `RESULTS_phase4.md` itself. A finding sitting correctly in one document
  and incorrectly in another is a different failure than not having found
  it — worth distinguishing when auditing multi-document, multi-session work,
  since the fix (propagate) is different from the fix for a fresh error
  (re-derive).

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
- **§6.1** — **error found and corrected, then narrowed after chat-side
  review** (2026-07-27, second pass): the "sorts monotonically ... on both
  rungs" claim is false at *raw* ppl@1024 in the mean (D0 < D1 < S, not
  D0 < S < D1) — verified against the same three variants' `summary.json`
  files. Narrowed on review: (1) this disagrees with §1's normalized-
  extrapolation metric, which orders S ahead of D1 (5.78× < 6.62×) — the two
  metrics answer different questions and both are correct; (2) the raw gap
  is ~2.5% with no reported ppl spread (only val loss has one in the grand
  summary), so "S is worst" is not a robust claim on its own — the robust
  claim is that width-monotonicity fails at ppl@1024, full stop.
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

### 12.1 Second pass (2026-07-27, same day): BCDI eigenvalue theorem

Chat side proposed a closed form for `cond(L_v)` (BCDI 2009, arXiv:0905.2987,
Cor. 7.3/Prop. 3.10) as a blocking check on §8.2's Monte-Carlo caveat, plus
four conditional re-derivations and a new Stage-0 statistic. All verified by
direct computation against `structure_tensor()`, not by reading the cited
paper (not fetched this pass).

- **Blocking check: PASS, after one false start.** First attempt used
  `u,w = v[:8], v[8:]` (the full CD-doubling halves) and got a large
  mismatch (eigenvalue error 0.64) — traced to the wrong split, not to a
  flaw in the derivation: **Koebisu Thm 3.9 fixes the component
  convention** (`‖u‖² = Σ_{i=1..7} aᵢ², ⟨u,w⟩ = Σ_{i=1..7} aᵢaᵢ₊₈`, i.e.
  `u,w` are the **imaginary** octonion halves, `u = v[1:8]`, `w = v[9:16]`,
  dropping each half's real component), which the first attempt lacked.
  Cite Thm 3.9 alongside Cor. 7.3 wherever this closed form is used. With
  the correct split: eigenvalues of `L_vᵀL_v` match the predicted
  `{1×8, (1+S)×4, (1−S)×4}` to 3.6e-15, and
  `cond(L_v) = √((1+S)/(1−S))` matches the SVD-computed condition number to
  9.4e-12, over 5000 random unit v. See §8.2 for the updated caveat.
- **Item 2(a)** (`D₂ = ‖v‖⁴(1−S²)`, cross-check against Koebisu's det
  formula already in `HANDOFF.md` §3.3) — confirmed algebraically: Koebisu's
  `D₂ = ‖v‖⁴ − 4(‖u‖²‖w‖² − ⟨u,w⟩²)` combined with the now-verified
  `S(v)` gives `D₂ = ‖v‖⁴(1−S²)` exactly. Not written up as a Koebisu
  novelty claim, per instruction.
- **Item 2(b)** (`min` over unit Q of r² `= (1−S(P))·|P|²`, attained on
  `Eig_{1−S}(P)`) — a direct Rayleigh-quotient consequence of the verified
  eigendecomposition; confirmed numerically against exact `eigh` output on 5
  random P (predicted value matches the true minimum eigenvalue exactly at
  each).
- **Item 2(c)** (`V₂(ℝ⁷) = G₂/SU(2)`) — not independently verified (the
  source addendum wasn't available locally to check its exact statement),
  but dimensionally self-consistent with everything else confirmed this
  pass: `dim V₂(ℝ⁷) = 7·2 − 3 = 11 = dim G₂ − dim SU(2)`, and the `S(P)=1`
  locus (zero-divisor-capable P) decomposes as an 11-dimensional base
  (normalized `(u,w)` pairs with `Re=0, ‖u‖=‖w‖, ⟨u,w⟩=0` — this project's
  own prior ZD characterization) times a 3-sphere of null-eigenspace
  directions, `11+3=14`, matching item 2(d)'s independently-measured
  dimension. Flagged as plausible-and-corroborated, not confirmed against
  source.
- **Item 2(d)** (does the repo's "dimension 14" mean the *normalized*-pair
  variety?) — **confirmed by direct computation, not by re-reading old
  doc text.** Built the Jacobian of `F(x,y)=x⊛y` at the known exact pair
  `(e₃+e₁₂, e₅+e₁₀)`, restricted to the tangent space of `S¹⁵×S¹⁵`: ambient
  30, rank 16, local dimension **14** — matches every "dimension 14"
  citation in this repo. For contrast, the *unnormalized* variety in
  `ℝ¹⁶×ℝ¹⁶` (no unit constraint) has dimension **16**, a genuinely
  different number. `FINDINGS_zd_variety_characterization_2026-07-24.md`
  §3.2's own method (ambient 30, rank of Jacobian) already measured the
  normalized object; this independently confirms it rather than just
  trusting the write-up.
- **Item 5's proposed statistic (eigenvalue multiplicity signature)** — ran
  it. S shows the `(8,4,4)` signature at **200/200** sampled points, no
  exceptions (theorem-fixed, as expected). X shows **16 distinct
  eigenvalues (no degeneracy at all) at 200/200** sampled points, all three
  grid seeds. A clean, non-overlapping binary discriminator — recommended
  over rank-deficiency for Stage 0 (see chat exchange, item 5). **Caveat
  carried into §12.2: this run sampled unit v from the free sphere
  (`rng.normal`, unit-normalized), not real model activations, and did not
  log full spectra — only multiplicity-partition counts. See §12.2 for the
  real-init version.**
- **Item 4(a) verified, with a correction of its own**: Koebisu
  (arXiv:2512.13002) and Biss–Dugger–Isaksen are indeed already in
  `PRIOR_ART_REVIEW_zda.md` §3 (confirmed at lines 188/190–192/476, not
  188/190–191/475 as claimed — a minor line-number drift, not a
  substantive error). One correction to the correction: the "40 variety
  points" closed-form claim it asks to fix in "`PHASE5_PLAN.md` §3.4" has
  no such section — `PHASE5_PLAN.md` has no §3.4 and no closed-form-locus
  text anywhere in it. That content lives in
  `SESSION_CONTINUATION_HANDOFF_2026-07-26.md` §3.4 instead, already
  committed, already hedged there as "speculative... a note under G4, not
  a directive." Addressed by addendum in that file rather than editing
  `PHASE5_PLAN.md`, which has nothing to correct on this point.
- **Item 2(d) also closes item 4(b)'s premise**: since `min r²` for unit P
  is now known in closed form (item 2(b)) and S's true zero divisors are
  already known exactly (336 pairs, `sedenion_kernel.py`), "can S reach
  r²=0" is a closed, proved fact for S, not an open question needing a
  restart search. Stage 0 item 3's global-infimum check should be rescoped
  to X only, as chat side proposed.

### 12.2 Third pass (2026-07-27, same day): real-init can't-vs-doesn't test

Chat side asked whether Stage 0's can't-vs-doesn't question might already be
answered from data on hand, via `(a)` were full spectra logged at item 5's
200 points, `(b)` confirm the X-side operator is the plain Gram form (no
conjugation), `(c)` compare the achievable floor against measured r² at the
same P, `(d)` state the sampling distribution.

**Answers:** `(a)` No — item 5's run only counted multiplicity-partition
tuples and printed one illustrative spectrum per seed; nothing was logged
to a file, and no run paired a spectrum with a measured r². `(b)` Confirmed
— every eigendecomposition in this pass (S and X alike) used the plain Gram
matrix `L_vᵀL_v` via a bare `torch.einsum`/`numpy.einsum` contraction, no
conjugation anywhere; this is valid for any bilinear map, including the
non-algebra shuffled tensor. `(c)`/`(d)` — not answerable from existing
data (nothing on disk pairs a floor with a measured r²), **but cheaply
answerable without a restart search or checkpoints**, so ran it fresh:
reused the exact real-init methodology from Stage 0 item 2's N2 check
(`phase4_init_r2_check.py` — real `K3Attention`, real `wq`/`wk`, real
`R_8` rotation, real grid dims d_model=384/heads=6/ctx=256, fresh untrained
weights, **not** free-sphere sampling — this is deliberate, since free
sampling is already known from N2 to badly mispredict r² at real init,
`RESULTS_phase4.md` §8.1 history). For 200 sampled causal query points per
variant (5 inits × 40 points), computed both the achievable floor
(min eigenvalue of `L_qᵀL_q` at the unit-normalized, *rotated* query — the
same vector the real score actually uses) and the actual minimum r² among
the real keys present at that position in the same batch:

| variant | median actual min r² | median achievable floor | median ratio | frac within 2× of floor |
|---|---|---|---|---|
| S | 0.5575 | 0.2284 | 2.32× | 40.0% |
| X seed1337 | 0.3747 | 0.0015 | 246× | 0.0% |
| X seed1338 | 0.3787 | 0.0016 | 206× | 0.0% |
| X seed1339 | 0.3653 | 0.0018 | 199× | 0.0% |

**At fresh, untrained init**, S's real queries already land within a
factor of ~2 of their theoretical best against the keys actually present
40% of the time — random projection alone gets respectably close. X's
achievable floor is ~100–150× lower than S's (consistent with the
~450–500× free-sphere accessibility gap already on record), **but the
actual keys present come nowhere near it — 0/200 within even 2× of the
floor, every seed**, and the *actual* r² achieved is comparable in
magnitude to S's (0.37–0.38 vs 0.56), not anywhere near X's much lower
floor.

**Reading, with the caveat stated plainly:** this is an **init-time**
result — fresh random weights, no training, no gradient signal — so it
does not by itself settle whether *training* declined an available
descent (the genuinely blocked "local" question, §10, still needs the
Drive-only checkpoints). What it does establish: X's dramatically more
accessible floor is not something random projections stumble into by
chance either — reaching it, if it happens at all, would have to be
something training actively finds, not a free byproduct of initialization
the way S's much smaller floor advantage partly is. This sharpens the
"doesn't" reading (§8.1's non-monotonic, flat-after-one-move X trajectory)
without resolving can't-vs-doesn't outright, and it was obtained without
the restart search chat side asked to hold off on.

### 12.3 Fourth pass (2026-07-27, same day): source documents received;
annihilator dimension confirmed as a second S-vs-X discriminator

`PRIOR_ART_addendum_2026-07-27.md` (rev. B) and `EIGENTHEORY_findings_2026-07-27.md`
arrived after §12.2 was written (receipt confirmed by reading both in full).
Everything in them that overlaps §12.1/§12.2 matches exactly — same `S(v)`
formula, same eigenvalue theorem, same `r²_min` Rayleigh-quotient result, same
recommendation to supersede rank-deficiency with the multiplicity signature.
Two genuinely new, independently-checkable claims, both verified:

- **Koebisu Thm 3.9 fixes the component convention** that cost the §12.1
  false start (`u,w` as the imaginary octonion halves) — already folded into
  §12.1 and the addendum in `SESSION_CONTINUATION_HANDOFF_2026-07-26.md`
  §3.4.
- **Exact-zero-divisor spectrum collapse (BCDI Prop. 7.4).** At an exact
  zero divisor, the spectrum should collapse to `{0×4, 1×8, 2×4}`.
  **Confirmed at 20 distinct exact zero-divisor elements of S** (`e_i±e_j`
  pairs, `structure_tensor()`), spectrum exact at every one — not a bound,
  realized exactly, every time.
- **Annihilator dimension as a Stage-0 discriminator (`PRIOR_ART_addendum`
  §4/§7).** For S this is now a proved fact (4, from the theorem, confirmed
  above). **For X: measured directly** — found near-zero-divisor points for
  all three grid seeds via Adam optimization (σ_min ≈ 1e-16, essentially
  exact), then read off the null-eigenspace dimension at each: **exactly 1,
  every seed**, next-smallest eigenvalue clearing zero by 0.012–0.019 with
  no ambiguity at any tolerance from 1e-3 to 1e-6. Matches the "generic
  bilinear map" prediction exactly. **A second, independent binary
  discriminator: S's annihilator is 4-dimensional, X's is 1-dimensional,
  both confirmed by direct measurement, not assumption.**

**One claim not independently re-derived**: `PRIOR_ART_addendum` §3.2's
`V₂(ℝ⁷) ≅ G₂/SU(2)` step (the G₂→S⁶→SU(3)→S⁵→SU(2) transitivity chain).
This is standard octonion/G₂ Lie theory (G₂ = Aut(𝕆) acts transitively on
unit imaginary octonions with stabilizer SU(3); SU(3) acts transitively on
unit vectors orthogonal to a fixed one with stabilizer SU(2)) and is
consistent with everything measured this pass (dimensionally: 11 = 14−3,
confirmed independently in §12.1 item 2(c) before this document arrived),
but it was affirmed from general mathematical knowledge, not re-derived from
BCDI/Koebisu's own text or independently proven here. Flagged, not treated
as repo-verified in the same sense as the numeric results above.

**Applied to the repo:** `PRIOR_ART_REVIEW_zda.md` §3 expanded with the BCDI
eigentheory paper (previously entirely absent) and Koebisu upgraded from
abstract-level to content-level citation, per the addendum's own diagnosis
that this was a citation-reading failure, not a search-coverage gap; new
§3.1(d) records the dimension-14 decomposition and both discriminators;
§8.4.7 gained item 4 recording the abstract-level-citation failure mode
itself. §8.2 above reframed: dismissing "dimension 14" as a discriminator
was the right call for the wrong reason: the shape (11+3 vs 14+0) is *not*
generic and now has a direct numeric handle in the annihilator dimension.

**Net: two corrections (§6.1, §8.1), one caveat (§8.2), everything else in
§3–§5, §7, and §8.3 exact — plus two new independently-confirmed
discriminators (eigenvalue multiplicity signature, annihilator dimension)
that did not exist in the original draft.** The document's headline claims
(H4a negative, H4c pass, the S–X double dissociation, wall-clock exceeding
its bound) are unaffected by any of this.
