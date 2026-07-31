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
neither is graded. **`k1 min` is an init-time statistic, not a trained
one** — `k1_row_spread` increases monotonically enough that its
minimum-over-the-entire-run equals the step-0 value in all 6 real runs
(verified directly against `p4_artifacts/*/eval_log.jsonl`; see §12.4(ii)),
so this column carries no information about the trained model.

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
4. **Can't vs doesn't — closed, both init and trained (2026-07-29, §12.5).**
   Whether X *can* reach low r² under the real parameterization was measured
   at init (§12.2) and corrected by a matched-N null control (§12.4): neither
   S nor X's real untrained keys beat chance, so the raw floor-proximity gap
   between them is target geometry (4-dim eigenspace vs. 1-dim needle), not
   steering. **The trained question is now answered too**, against real
   TinyStories val batches and the actual trained checkpoints (Drive access
   set up 2026-07-28/29), using two nulls: a naive i.i.d.-unit-vector one
   and a **correlation-matched** one (real, unrelated key windows — needed
   because real same-sequence keys are positively correlated, which biases
   the i.i.d. null; confirmed directly by the correlation-matched null
   recovering percentile≈0.5 at init for both variants, where the i.i.d.
   null does not). **Under the validated correlation-matched null, both S
   and X's trained keys beat chance at finding low-r² partners — S's effect
   is real and roughly 1.8× X's** (z≈−6.4 vs. z≈−3.6), not the order-of-
   magnitude gap the i.i.d. null alone suggested. **Both effects are highly
   significant and small in absolute size** — median percentile 0.41 (S)
   and 0.45 (X) against a no-steering expectation of 0.50, i.e. S beats
   ~59% of matched-correlation alternatives and X beats ~55%, not "always"
   for either. **Engagement alone therefore does not by itself account for
   the S-vs-X extrapolation dissociation (§4/§8) — both variants engage the
   mechanism to a real, measurable, and broadly similar degree; that link
   is open, not established, and no framing to the contrary should be
   carried forward.** This is the pre-registered rule's third branch, "both
   steer; compare magnitudes," not the second. See §12.5 for the full
   breakdown, both nulls side by side, and how the
   correlation-matched null also resolved (not just flagged) the session's
   one surprise — a real anti-chance skew for **S at init too** under the
   i.i.d. null, now confirmed to be exactly the correlation-penalty effect
   that motivated building the second null in the first place.
5. **Seed variance in the dense family is large** — D0p ppl@512 swings 46%
   between seeds — which sets a floor on readable margins for any dense
   comparison.
6. **A1/A2/A3 are post-hoc**, not blind (§7.2).

---

## 10. Open items

| item | cost | blocks | status |
|---|---|---|---|
| Constrained-infimum check (can't vs doesn't) | inference only | §8.1 framing | **CLOSED, both init and trained (2026-07-29)** — S's global infimum is a closed-form theorem (`EIGENTHEORY_findings_2026-07-27.md` §5); X's init-time accessibility was measured (§12.2) and null-corrected (§12.4) — neither variant beats chance untrained. **Trained-checkpoint question answered in §12.5**: real Drive-synced checkpoints (`paul@venicedispatch.info` account) + real TinyStories val batches, layer 0, under both an i.i.d. null and a correlation-matched null (the latter validated as unbiased by recovering percentile≈0.5 at init for both variants) — **both S and X's trained keys beat chance under the validated null, S's effect ≈1.8× X's**, decision-rule branch 3 ("both steer, compare magnitudes"). Trajectory (early/mid/late) checked and found unavailable — `phase4_grid.py` only ever kept a single rolling final checkpoint, no intermediates exist on Drive for any S/X run. |
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
- **A third instance of the same pattern** (chat-side observation,
  2026-07-31): §9 limitation 5's "seed variance in the dense family is
  large ... sets a floor on readable margins for any dense comparison" was
  already on record before `PHASE5_priorityA_crossing_2026-07-31.md`'s
  S-vs-D1 and Q0-ranking work, which independently re-derived essentially
  the same conclusion (via per-seed sign counts showing 2/1 and 1/2 splits
  on comparisons involving D0/D0p/D1/Q0) without citing it. The re-derivation
  itself was still the right verification to do — but a fact established
  correctly here sat uncited in a later document making exactly the claims
  it would have flagged in advance. Three instances is enough to treat this
  as a standing pre-write check: before drafting a new document's limitations
  section, grep prior documents for the same claim rather than re-deriving
  it cold.

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

### 12.4 Fifth pass (2026-07-27/28): matched-N random-key null (the
blocking item), plus two carried-question resolutions

Three items carried from the previous session's handoff, addressed in
order.

**(i) X-side operator confirmation.** Re-confirmed: §12.2(b) already
established that every eigendecomposition in this pass (S and X alike)
uses the plain Gram matrix `L_vᵀL_v` (a bare `einsum` contraction, no
conjugation anywhere). This is required for X specifically because BCDI's
`M_a := (1/‖a‖²)L_{a*}L_a` presupposes an algebra with a conjugation, and
the shuffled tensor is not one — the plain Gram form (squared singular
values of `L_v`) is the only operator that transfers. No new work needed;
restated here for the record since the question recurred.

**(ii) Grand-summary `k1 min` is not the same quantity as the r²/eigenvalue
work, and there is no training-driven reversal.** The handoff asked
whether `k1 min` (grand summary §3: S=1.33, X=2.42, described there as
"trained") is the same quantity/normalization as the cond(L_v)/r²-floor
work, since if so, "at init X is lower than S" while the trained figures
show S<X would be a genuine dynamic reversal worth writing up. Checked
directly against `p4_artifacts/*/eval_log.jsonl` (real logged data, not a
rerun): `k1_row_spread` is the per-eval **minimum-over-layers** of the
maximum absolute difference between raw attention-score rows at different
query positions against a shared causal key window (`phase4_train.py`
lines 105–110) — a K1-degeneracy liveness guard measured in raw-logit
units, not the algebraic Rayleigh-quotient/eigenvalue quantities of §12.1–
§12.3. It is a different object by construction, not merely a different
normalization of the same one.

Directly checked all 6 real runs' full eval trajectories: for **every
one** (S and X, all 3 seeds), the global minimum of `k1_row_spread` over
the *entire* training run occurs at **step 0**, not at the end:

| run | step-0 (=min) | step-end (max) |
|---|---|---|
| S seed1337/1338/1339 | 1.3426 / 1.3316 / 1.3309 | 2.8436 / 2.8989 / 3.1617 |
| X seed1337/1338/1339 | 2.7031 / 2.6189 / 2.4202 | 5.6101 / 5.4492 / 5.8030 |

`k1_row_spread` **increases** over training for both variants (min always
at init, max always at the end) — the opposite of monotonically
decreasing, so `k1_guard_min` (`min over all evals`, per `phase4_grid.py`
line 714) always equals the init value in every one of these 6 runs, and
the grand-summary table's "k1 min" column (S=1.33, X=2.42, matching
seed1339's init value for each) **is an init-time statistic, not a
trained/end-of-run one** — the "(trained)" label in the handoff's framing
of the question was incorrect. There is consequently **no dynamic
reversal to report**: at init, X's `k1_row_spread` (2.42–2.70) is already
*higher* than S's (1.33–1.34), matching the grand-summary ordering
exactly, not lower. The "at init X is lower than S" premise instead
describes a *different* quantity — the r²-floor numbers in §12.2, where
X's achievable floor (0.0012–0.0022) genuinely is far lower than S's
(0.207–0.228) — and conflating the two was the source of the apparent
puzzle. No write-up change needed to the grand summary; flagging this
here so the conflation doesn't recur.

**(iii) median-of-ratios vs ratio-of-medians, resolved retroactively for
§12.2 without needing the lost per-point data.** §12.2's script was not
saved to the repo, so its exact "median ratio" column (S: 2.32×) can't be
recomputed directly from raw per-point values. But it can be checked for
internal consistency: `ratio_of_medians` from the two medians §12.2
itself reports is `0.5575 / 0.2284 = 2.44`, not the `2.32` printed in the
table. Since these disagree, **§12.2's "median ratio" column must have
been `median_of_ratios` (median taken per-point, before aggregating), not
`ratio_of_medians`** — an internal check that resolves the ambiguity
without the original script. Going forward (this pass and any rerun),
both conventions are logged explicitly (see table below) rather than a
single ambiguous "ratio" column.

**The blocking item: matched-N random-key null control.** New script
`phase4_matched_N_null.py`, same real-init methodology as §12.2 (real
`K3Attention`, real `wq`/`wk`, real `R_8` rotation, real grid dims
d_model=384/n_heads=6/ctx=256, fresh untrained weights, 5 inits × 40
causal query points = 200 points/variant). At each sampled point `P`, in
addition to the real min r² among the `N` real keys present at that
causal position (`N` = position+1) and the closed-form achievable floor,
drew `N` freshly-sampled random unit keys (100 repeated trials per point,
median taken) and computed `ratio_null = actual_min_r² / random_N_min_r²`
— matched-N so a thin target isn't penalized just for being hard for
*anyone*, real or random, to hit:

| variant | median actual | median floor | median random-N-min | ratio_of_medians | median_of_ratios |
|---|---|---|---|---|---|
| S | 0.5617 | 0.2071 | 0.5358 | 1.048 | 1.002 |
| X seed1337 | 0.3649 | 0.0022 | 0.3678 | 0.992 | 1.001 |
| X seed1338 | 0.3753 | 0.0012 | 0.3731 | 1.006 | 0.993 |
| X seed1339 | 0.3624 | 0.0018 | 0.3680 | 0.985 | 0.996 |

Every variant, every seed: `ratio_null ≈ 1` (0.98–1.05 both conventions),
and 99.0–100% of individual points fall within [0.5×, 2×] of the matched-N
null. **Applying the pre-specified three-way decision rule: both S and X
land in the "both ≈ 1" branch — neither variant's real (untrained, freshly
projected/rotated) keys beat chance at finding a low-r² partner, relative
to N random draws matched for the same N.**

**This means §12.2's headline framing needs a correction, not to its
numbers but to what they were read as showing.** S landing near its floor
40% of the time while X lands near its floor 0% of the time is **not**
evidence that S's real keys are "steered" toward low r² more than chance
— matched-N random draws land just as close to S's floor as S's real keys
do (median random-N-min 0.536 vs actual 0.562), and just as far from X's
floor as X's real keys do (median random-N-min 0.368–0.373 vs actual
0.362–0.375). The entire §12.2 gap is explained by **target width alone**:
S's floor sits on a 4-dimensional eigenspace (BCDI multiplicity-4
annihilator direction), reachable by chance a meaningful fraction of the
time; X's sits on a ~1-dimensional needle (measured annihilator dimension
1), unreachable by chance regardless of any steering effect on either
side. At **fresh random init**, this is exactly the expected result —
`wq`/`wk` are literally untrained random projections, so "real" keys and
random unit draws should behave identically until training does
something with them, and this control confirms they do.

**Consequence for the can't-vs-doesn't question (§3.1/§10):** §12.2's
closing line ("sharpens the 'doesn't' reading... without resolving
can't-vs-doesn't outright") **overclaimed** — corrected here, not
struck through. The init-time floor-proximity gap does not sharpen
"doesn't" at all once the matched-N null is controlled for; it is fully
explained by geometry. **The only evidence actually on record for
"doesn't" remains Stage 0 item 1's *training-dynamics* finding**
(`PHASE5_stage0_findings_2026-07-26.md` §1: X's log-slope goes flat after
one early move while S sustains negative log-slope through nearly the
whole 18,311-step run) — a genuinely dynamic, trained-model result,
untouched by this correction. The checkpoint-dependent "local" question
(§10, still blocked on the ~3GB Drive-only checkpoints) remains the only
way to test whether *trained* keys show the steering this init-time
control finds absent at step 0.

**Floor discrepancy, §12.2 vs §12.4, diagnosed.** S's median achievable
floor moved 0.2284 (§12.2) → 0.2071 (§12.4, +9.3% relative), while median
actual min r² barely moved (0.5575 → 0.5617, +0.75%) — asymmetric, so not
generic sampling noise across the board. Checked in the specified order:

- **(a) Unit-normalization.** `achievable_floor` is *not* pre-normalizing
  `P`; it divides by `‖P‖²` inside the Gram matrix (`M = AᵀA/‖P‖²`). Tested
  directly: `achievable_floor(T,P)` vs `achievable_floor(T,P/‖P‖)` over 500
  random-scale draws — identical to 1.2e-12. Normalization is handled
  correctly; not the source.
- **(b) Same 5 inits?** Yes — `torch.manual_seed(1000+init_seed)` for
  `init_seed in range(5)`, identical convention to `phase4_init_r2_check.py`
  (which §12.2 states it reused) and to `phase4_matched_N_null.py`. Same
  grid dims (384/6/256) mean the model weights and the input batch `x` are
  bit-identical draws across all three scripts for each init. Not the
  source.
- **(c) Theory tiebreak.** `achievable_floor`'s eigenvalue computation
  matches the closed-form `1−S(P)` (`EIGENTHEORY_findings_2026-07-27.md`
  §2.1) to **4.5e-12** over 500 synthetic vectors and **1.9e-7** over 200
  real rotated query vectors — the formula is correct, gated exactly as
  specified, ruling out a math error in either script's floor computation.

**Actual source: point-sampling variance, not a bug.** `S(P)` (hence the
floor) depends on which specific query direction gets sampled and is far
more sensitive to the exact 200 points drawn than the "actual min r²"
statistic is (that one aggregates over many real keys per point, damping
point-to-point variation; the floor is a single number per point with no
such averaging). Confirmed directly: resampling with a "clean" point-only
RNG stream (no interleaved random-null-trial draws consuming the stream
first, unlike `phase4_matched_N_null.py`'s `run_variant`, which interleaves
100 null-trial draws per point between successive `sample_points` calls)
gives median floor **0.2238** — 2% from §12.2's 0.2284, not 9%. The
interleaving pattern in `phase4_matched_N_null.py` is not a bug (every
draw is still a valid, independently-sampled query point), but it does
mean that script's specific 200-point sample lands further from §12.2's
than a fresh resample would. **Net: both scripts' floor figures are
correct Monte Carlo estimates of the same underlying quantity; they differ
because 200 points is not enough to pin the median of `S(P)`'s
distribution tighter than roughly this range.** Does not affect anything
load-bearing: `ratio_null` never uses the floor value, and the 4-dim-vs-
1-dim geometric argument is a structural fact (BCDI/annihilator-dimension
theorems), not a Monte Carlo estimate.

### 12.5 Sixth pass (2026-07-28/29): matched-N null against real
TinyStories batches AND trained checkpoints — Phase 5 item 1 resolved

Drive access to the actual grid checkpoints was set up this session (they
turned out to live under a second Google account, `paul@venicedispatch.info`,
not the one already mounted locally — `MyDrive/p4_runs_ts/` and the matching
tokenized data cache `MyDrive/zda_data_cache/`, both synced locally to
`p4_checkpoints/` and `p4_val_data/`, gitignored). New script
`phase4_matched_N_trained.py` supersedes `phase4_matched_N_null.py`'s
methodology outright, per five chat-side specification items, checked
before building rather than assumed:

**1. Input: real data, not `torch.randn`.** Checked first, as required: both
§12.2 and §12.4 used synthetic Gaussian input, not real TinyStories
batches — trained `wq`/`wk` on synthetic input answers a question about a
distribution training never touched. **Not comparable as-is.** Fixed by
rebuilding *both* the init and trained conditions on the identical real-data
pipeline (`TokenDataset` over `p4_val_data/val.bin`, the same class
`phase4_grid.py` trains against) — so this pass's `init` numbers supersede
§12.2/§12.4's for comparability, not just an addition alongside them.

**2. Norm-matched null.** Checked analytically and numerically before
deciding whether to change the null construction: the model's own
`r2 = ‖P⊛Q‖²/(‖P‖²‖Q‖²)` (`phase4_layers.py`'s actual score formula) is
**exactly scale-invariant in `‖Q‖`** (and `‖P‖`) by construction — verified
to 1.9e-10 relative difference under 0.01×–100× rescaling of `Q`. Drawing
null keys as unit vectors therefore introduces no bias relative to real
(non-unit) keys regardless of norm drift; no renormalization was needed.
Per the spec's own fallback ("log the key-norm distribution... if S and X
differ substantially, that is a finding on its own") — they do: pooled
across 3 seeds, key norms grow from init to trained by **+5.2% for S**
(median 2.245→2.362) vs. **+13.9% for X** (median 2.237→2.550), a real,
∼2.7× larger relative drift for X. Confirmed inert to the r² metric itself,
reported as a standalone descriptive fact.

**Follow-up question (chat-side review, 2026-07-29): is this normalization
in the forward pass, or only in the measurement code?** If it lived only
in this analysis script and not in `K3Attention` itself, the key-norm
drift above would act on the score the *real model* actually optimizes
while being invisible to this control — a real gap. **Checked directly
against `phase4_layers.py`: it is in the forward pass, unambiguously.**
`K3Attention.scores()` (lines 104–126) computes `den = (q.pow(2).sum(-1)...
* k.pow(2).sum(-1)...)` and `r2 = num / den` — this is the *same* method
`forward()` calls (`_, s = self.scores(x, pos_offset)`) to produce the
score `s = -gamma * r2` that gets soft-maxed into the real attention
weights, both during training and in this script's own extraction (which
reads `attn.wq`/`attn.wk`/`attn._rotate` directly, not a separate
reimplementation). There is no unnormalized code path anywhere in the
model — `scores()` *is* the forward pass's score computation. So the r²
this control measures is exactly what the model trains under and what its
attention weights are computed from; the key-norm drift is real and
correctly logged as inert to r² specifically because the model's own
normalization (not this script's) already divides it out at every
application, training included.

**3. Stratified by N-quartile, not pooled** — see the tables below. This
item's spec also carried a proposed second null (see "correlation-matched
null" below), which turned out to matter more than the stratification
itself.

**4. Trajectory (early/mid/late checkpoints) — checked, unavailable.**
`phase4_grid.py`'s checkpoint save (`save_ckpt_atomic`) writes a single
rolling `ckpt_last.pt` at every eval, overwritten each time, saved as the
final `ckpt.pt` at completion. Confirmed against Drive directly: every
`p4_runs_ts/{variant}_seed{seed}/` has exactly one `ckpt.pt`; the only
other checkpoint anywhere (`p4_runs_archive/D0p_seed1339_partial_20260722`)
is an unrelated variant's crash-recovery snapshot. **No intermediate
trajectory is recoverable for S or X, at any seed.** Not fabricated or
substituted — flagged as a real limit on what this pass can settle (see
"what this does and doesn't resolve" below).

**5. Empirical percentile readout**, alongside both ratio conventions
(§12.4(iii)): for each point, the fraction of its own matched-N null
trials at least as low as the actual real-key minimum. Under no steering
this is Uniform(0,1); low values mean real keys beat chance (steering
toward low r²), high values mean real keys land worse than chance. **Trial
count raised 100→300** (chat-side review, 2026-07-29 — 100 trials put
extreme points near the percentile resolution floor, where "just below
all draws" and "far below all draws" both read as ~0.00–0.03 and can't be
told apart) and a **floor-immune magnitude statistic** added alongside the
rank-based percentile: `ratio_p5 = actual_min / (5th percentile of the
null draws)`, which doesn't saturate the way a rank does. For trained S
under the i.i.d. null, `ratio_p5` ≈ 0.92–0.95 (pooled median 0.935) —
consistent with a true rank of roughly 2–5%, confirming the percentile≈0.03
reading is not a floor artifact of the trial count.

**Layer 0 only** (matches every other headline number in this project —
the r²_min@0/@end table, PHASE5_stage0's early-lock trajectory, most of
H4c's descending heads). Checkpoint loading verified exactly: loaded
`wq.weight` matches the raw checkpoint tensor to 0.0 absolute difference;
a fresh untrained init differs from it by 0.15 max absolute difference —
confirms the "trained" condition is really trained, not silently falling
back to a fresh init on a load failure.

**The correlation-matched null (chat-side review item 3, 2026-07-29) — the
single most consequential addition this pass.** The i.i.d. unit-vector
null treats the N real keys at a causal position as if they were N
independent draws. They aren't: keys at nearby positions in one real
sequence, through one set of shared weights, are positively correlated.
**The minimum of N positively-correlated draws is stochastically *larger*
(less extreme) than the minimum of N i.i.d. draws with the same marginal
distribution** — correlated samples cluster instead of spreading out to
explore the tails independently. So *absent any steering*, real keys are
expected to land *worse* than the i.i.d. null, and a percentile above 0.5
is the **correct null expectation**, not a defect. This is exactly the
account offered (as a plausible, unconfirmed guess) for `init S`'s
above-0.5 skew below — and it is now directly testable.

Built a second null: instead of N independent random unit vectors, draw N
real keys from an **unrelated (batch, offset) window of the same length**
— a genuine, naturally-correlated block of real keys with no relationship
to this specific query (different story, same absolute causal positions
so the same rotation angles apply). Under this null, percentile 0.5 *is*
the correct no-steering reference directly, with no correlation penalty to
correct for.

**Result: at init, the correlation-matched null recovers percentile ≈ 0.5
almost exactly, for both variants, all three seeds** (pooled `init S`:
mean pct 0.497, z=−0.25; `init X`: mean pct 0.495, z=−0.43 — both fully
consistent with pure noise). **This directly confirms the correlation-
penalty explanation** — it is not a guess anymore. The i.i.d. null's
init skew (S: z=+10.0, X: z=+3.5) is now understood precisely: it is the
correlation penalty, present at both variants, larger for S than X
(plausibly because S's exact-norm-preserving `R_8` rotation induces more
regular structure among nearby real keys than X's rotation, which is
documented elsewhere in this project — `phase4_X_positional_check.py` —
to *not* preserve norms or the shared-phase null cleanly; not chased
further here).

**Pooled results, both nulls (3 seeds × 200 points = 600 per condition):**

| condition | i.i.d. z | i.i.d. median pct | corr-matched z | corr-matched median pct |
|---|---|---|---|---|
| init S | +10.0 | 0.672 | **−0.25** | **0.490** |
| init X | +3.5 | 0.573 | **−0.43** | **0.495** |
| trained S | −24.5 | 0.030 | **−6.4** | **0.405** |
| trained X | −5.8 | 0.378 | **−3.6** | **0.450** |

† z treats the 600 points as independent for a rough gauge — they aren't
fully (points share batches/models), so this overstates formal
significance and should be read as a descriptive ranking, not a validated
p-value; the *init* rows above (landing at |z|<0.5 under the corr-null
against a construction that should be unbiased) are themselves a rough
empirical check that this approximation isn't wildly miscalibrated.

**This revises the trained reading, not just the init one.** Under the
i.i.d. null, trained X looked close to noise (small pooled deviation,
concentrated at small N — see below). Under the correlation-matched null
— the one now validated as unbiased at init — **trained X shows a real,
consistent, three-seeds-agree deviation (z=−3.6, ~55% of matched-
correlation alternatives beaten), smaller than S's (z=−6.4, ~60% beaten)
by roughly a factor of 1.8, not the 4–9× gap the i.i.d. null suggested.**
Both are now `< 0.5`. This is the pre-registered rule's **third branch —
"both steer; compare magnitudes"** — read directly off the more rigorous
instrument, not forced into the second branch as the first draft of this
section did.

**Effect size, stated explicitly alongside significance (chat-side review,
2026-07-29) — these are highly significant but small effects, and neither
number alone should stand in for the other.** Under the correlation-
matched null, median percentile is **0.41 for S and 0.45 for X, against a
no-steering expectation of 0.50** — absolute deviations of 0.09 and 0.05.
`z=−6.4` and `z=−3.6` describe how *reliably* those small deviations recur
across 600 pooled points (and, per the caveat above, likely overstate
formal significance further since the points aren't fully independent);
they say nothing about *how far* any single real key typically sits from
a typical unrelated one. Reading percentile 0.41 concretely: **S's real
keys beat about 59% of matched-correlation alternatives, not "always" or
"by a wide margin."** X's beat about 55%. Both are real, reproducible
departures from chance — and both are modest ones. A z-score this large
from an effect this small is a function of pooling 600 points, not a
license to describe either effect as large.

**N-quartile breakdown, both nulls (pooled 3 seeds):**

| condition | Nq1 | Nq2 | Nq3 | Nq4 |
|---|---|---|---|---|
| trained S, i.i.d. median pct | 0.06 | 0.03 | 0.02 | **0.02** |
| trained S, corr median pct | 0.33 | 0.44 | 0.44 | 0.41 |
| trained X, i.i.d. median pct | **0.31** | 0.43 | 0.37 | 0.44 |
| trained X, corr median pct | 0.37 | 0.44 | 0.41 | 0.38 |

Under the i.i.d. null, **S's effect is flat-to-strengthening across the
entire N range** — the median ratio if anything gets more extreme at high
N (0.687→0.598 across quartiles) — the opposite of the "min-over-N
mechanically converges to the floor for everyone" artifact the spec warned
about, and the signature of a real, population-wide effect rather than one
lucky key at small N. X's i.i.d.-null profile is less clean (strongest at
Nq1, closer to null at Nq2–4).

**Correction to the first draft's reading of that pattern (chat-side
review, 2026-07-29): "fades toward null at large N" does not by itself
prove artifact.** The same min-over-N convergence that makes S's flat
profile diagnostic of a real effect would *also* erode a weak-but-genuine
effect at large N — the convergence is symmetric and doesn't distinguish
"never real" from "real but too weak to survive the squeeze." Under the
correlation-matched null, the N-profile for both S and X is noisier and
does not show S's clean flat/strengthening pattern either (200 trials
drawn from a ~300-real-window pool, smaller effect sizes throughout) — so
the N-stratification argument, on its own, is weaker evidence than the
first draft treated it as. **The correlation-matched null's pooled result
is now the primary basis for "X steers too, less than S," not the
N-profile**, which is presented here as a secondary, i.i.d.-null-only
diagnostic rather than independent proof.

**Chat-side prediction, recorded before the run: "S below 1, X near 1 —
the 'doesn't survives' branch. Moderate confidence."** Right on direction
(S's effect is real and larger than X's, confirmed under the more
rigorous null too) and right to flag only moderate confidence. **Not borne
out at face value**: under the correlation-matched null — the one the
init check validates as the fair reference — X is not `≈ 1` (`≈0.5` in
percentile terms); it shows a real, reproducible, smaller effect. The
recorded "live alternative" ("both near 1 even trained... H4c's descent
doesn't translate into keys landing nearer the floor than chance") is
closer to wrong for both variants than either extreme reading — **the
result that actually obtained is a third option neither framing stated
outright: both variants steer, real and reproducible for both, S roughly
1.8× X.**

**The `init S` skew: no longer a "plausible, unconfirmed" surprise — now a
confirmed, understood one.** The correlation-matched null recovering
percentile ≈0.5 at init for both variants directly confirms the
correlation-penalty account above; it is not the `phase4_init_r2_check.py`
"real embeddings don't resemble free-sphere sampling" mechanism this
section originally guessed at (that mechanism is still true and relevant
elsewhere in this project, just not the explanation for *this* number).
The i.i.d. null remains useful — its own before/after delta (S: 0.67→0.03;
X: 0.57→0.38) still shows the same qualitative pattern — but the
correlation-matched null is the more defensible number to lead with going
forward, since it needs no delta workaround: its own zero point is
directly interpretable.

**Caveat on `init S`'s "3 seeds," caught on review, not by an outside
check: they were not 3 independent random inits in the first version of
this script.** It originally seeded the init-mode model by
`1000+batch_seed` (`batch_seed` ∈ 0–4), never by the outer `seed`
(1337/1338/1339) loop variable — so for **S specifically**, whose
construction has no seed-dependence at all (unlike X, whose structure
tensor `shuffled_structure_tensor(seed)` genuinely differs per seed even
though its `wq`/`wk` weights don't), all three `init S` "seed" rows shared
*bit-identical* `wq`/`wk`/`wv`/`wo`/`gamma` — only the real-data batch and
point sample differed. **Fixed** (now `seed*10+batch_seed`, giving
genuinely independent inits per seed) **and rerun** — the numbers above
are from the corrected version; the anti-chance skew held up under
independent inits (i.i.d. pooled z moved from +10.3 to +10.0, unchanged
within noise), so this was a real methodological gap, not one that changed
the finding. Does not touch the trained comparison (S and X trained each
come from 3 genuinely independently-trained checkpoints, unaffected by
this bug either way).

**What this does and doesn't resolve.** It answers the "trained" half of
can't-vs-doesn't for the *final* state, under the now-validated
correlation-matched null: **both S and X's trained keys beat chance at
finding low-r² partners; S's effect is real and roughly 1.8× X's, not an
order of magnitude larger as the i.i.d.-null-only reading first
suggested.** It does not distinguish "X steered early (Stage 0 item 1's
flat-after-one-move log-slope) and then stopped at a smaller magnitude
than S" from "X steered weakly the whole time" — that needs early/mid/late
checkpoints, confirmed unavailable (item 4). What it does add: X's final
state shows a real, if modest, steering signal — consistent with *some*
version of Stage 0 item 1's early move having been genuine, just smaller
and/or less sustained than S's, rather than the "X never steered at all"
reading the first draft of this section leaned toward.

**Files**: `phase4_matched_N_trained.py` (script, now with both nulls and
independent-init seeding), `p4_matched_N_trained_results.npz` (raw
per-point arrays, all 12 conditions, both nulls), `p4_checkpoints/` and
`p4_val_data/` (local copies of the Drive-synced checkpoints and tokenized
val set, gitignored, re-fetchable from `MyDrive/p4_runs_ts` +
`MyDrive/zda_data_cache` on the `paul@venicedispatch.info` Drive account).

### 12.6 Seventh pass (2026-07-29): scope-check the init-seeding bug against
the multiplicity-signature and annihilator-dimension runs

§12.5's fix (`init S`'s three "seed" rows sharing bit-identical weights,
caused by seeding init-mode model construction as `1000+batch_seed`,
independent of the outer grid-seed loop) raised the question of whether the
same bug shape affects two other "all three seeds" claims in this document:
the eigenvalue multiplicity-signature run (§12.1: S `(8,4,4)` at 200/200
points; X 16 distinct eigenvalues at 200/200 points, "all three grid
seeds") and the annihilator-dimension run (§12.3: S theorem-fixed at 4; X
measured at exactly 1, "every seed").

**Confirmed: neither is affected, and the seed semantics are already
correctly stated, not merely correctly named by coincidence.** Both checks
operate directly on the structure tensor evaluated at random unit vectors
`v` (`rng.normal`, unit-normalized) — there is no `K3Attention`
instantiation, no `wq`/`wk` projection, and no `torch.manual_seed` call
anywhere in their construction, so the specific bug shape (a model-weight
seed accidentally independent of the outer loop variable) has no code path
to occur in. "Seed" means something different and unambiguous in each of
the three contexts, and the document's existing wording already tracks
this correctly:

- **§12.4/§12.5 (matched-N null, `phase4_matched_N_trained.py`):** "seed"
  indexes one of 3 *trained model checkpoints* (or, pre-fix, was supposed
  to index 3 independent fresh inits) — a genuine per-replicate weight
  identity, which is exactly what the bug accidentally collapsed for `init
  S`.
- **§12.1/§12.3 (multiplicity signature, annihilator dimension,
  `phase4_matched_N_null.py`/`phase4_init_r2_check.py` pattern): "all three
  grid seeds" indexes which of the 3 *actual* `shuffled_structure_tensor(seed)`
  constructions (seed ∈ {1337,1338,1339}) is under test — three genuinely
  different tensors, not three draws of a shared random process. There is
  no sense in which these three could have collapsed to "bit-identical" the
  way `init S`'s weights did, because the object being varied *is* the seed
  argument itself, consumed directly by `shuffled_structure_tensor()`, not
  laundered through an unrelated `torch.manual_seed` call.** For S, no
  seed loop is even meaningful — the `(8,4,4)` signature and annihilator
  dimension 4 are theorem-fixed for *every* nonzero `v`, confirmed
  numerically rather than sampled per seed.

No wording fix was needed in §12.1/§12.3's own text (both already say
"grid seeds" / "every seed" in a way that, read carefully, ties to the
X-tensor-construction seed rather than implying independent model-weight
replicates) — this section exists to make that reading explicit rather
than leave it as something a reader has to reconstruct themselves, and to
record that the scope-check was actually performed rather than assumed.
One residual, unchanged limitation carried over from §12.1: the "200
sampled points" for the multiplicity-signature run were never logged as a
committed script — this check was run ad hoc and only its summary
(multiplicity-partition counts, one illustrative spectrum per seed) survived
into this document, so it cannot be independently re-executed byte-for-byte
from the repo alone. That gap is orthogonal to the seeding-bug question
just closed.

### 12.7 Stage A pre-registration (2026-07-29, LOCKED BEFORE RESULTS)

Both S and X steer in-distribution (§12.5); the S-vs-X double dissociation
(§5) is an out-of-distribution phenomenon. APM Stage A
(`APM_STAGE_A_KICKOFF_2026-07-29.md`) measures steering directly on the
extrapolation-rung inputs (ctx=512/1024) using the same validated
correlation-matched null, same trained checkpoints, no new training. Per
that document's own STEP 4 instruction, the numeric definitions of
"holds" and "collapses" below were proposed and owner-approved **before
`phase4_matched_N_trained.py --rungs 512,1024` was run or any output
examined** — this section was written first, the run happens after it.

**Statistic** (identical to §12.5): per (variant, rung), pool the 3 seeds'
corr-matched-null percentiles (600 points, 3×200) and compute the median
percentile and `z = (0.5 − mean_percentile) / (std_percentile / √n)`
against the Uniform(0,1) no-steering null.

**Per-(variant, rung) call:**
- **HOLDS**: pooled `z ≤ −3` **and** each of the 3 seeds individually has
  its own median percentile `< 0.5` (not just the pooled figure — guards
  against one seed's large effect masking the other two showing nothing).
- **COLLAPSES**: pooled `|z| < 2`.
- **AMBIGUOUS** (`2 ≤ |z| < 3`): reported as its own category, not forced
  into either bucket. For the 3-way decision rule below, AMBIGUOUS is
  treated as COLLAPSES (the conservative direction — it withholds
  mechanism-paper support rather than grants it on a marginal result).

**3-way decision rule** (unchanged from the kickoff doc, made numeric):
- **Mechanism paper**: S is HOLDS at **both** rungs (512 and 1024) **and**
  X is COLLAPSES-or-AMBIGUOUS at **both** rungs.
- **Fallback paper**: every other outcome, with no exceptions carved out
  now. This explicitly includes a rung-split (e.g. S holds at 512 but not
  1024, or X holds at only one rung) — a clean dissociation at both
  extrapolation lengths is required for the mechanism reading; a partial
  or mixed result is not a fourth, more interesting story to interpret
  post hoc, it is fallback-paper evidence.

Sanity gate (unchanged from the kickoff doc): init-mode results at each
rung are computed and reported before that rung's trained results; if the
corr-matched null's init median percentile is not close to 0.5 at that
rung, the trained numbers at that rung are flagged as untrustworthy before
being read, not silently interpreted anyway.

### 12.8 V3b non-separability (2026-07-29): confirmed, closed as not-buildable

**The question.** V3b (conditioning-matched, algebra-broken control) was
reopened `2026-07-26/27` (`SESSION_CONTINUATION_HANDOFF_2026-07-26.md`
§3.2, `PHASE5_PLAN.md` §4) to attribute the S-vs-X dissociation between two
accounts: is it the zero-divisor algebra specifically, or merely the
~15–16× conditioning gap (`cond(L_v)`) between S and X, mediating a
shape-of-attention effect rather than an algebraic one? Building V3b
requires a tensor that (i) reproduces S's conditioning (the `(8,4,4)`
eigenvalue-multiplicity spectrum of `L_vᵀL_v`, BCDI 2009, §12.1) while (ii)
breaking the actual sedenion multiplication (different ZD variety, no
bilaterality — "algebra broken" in the same sense X already is). Flagged
as needing "real design work" and never built.

**(a) Confirm or refute: can (i) and (ii) coexist?** Tested two ways,
`phase4_v3b_rigidity_check.py`, no training, no GPU:

- **Rigidity.** The true structure tensor was perturbed by increasing-
  amplitude generic Gaussian noise (relative amplitude 0 to 1×, `‖noise‖=1`
  direction, `‖T0‖=16`). The `(8,4,4)` signature and near-zero intra-group
  eigenvalue split (2.7×10⁻¹⁵ at ε=0, exact to machine precision) hold at
  ε=10⁻⁸ and 10⁻⁶ (split ∝ ε, as expected for a first-order perturbation),
  then **the signature is already fully collapsed to 16 distinct
  eigenvalues by ε=10⁻³–10⁻²** — i.e. a generic perturbation at **0.1–1%**
  of the tensor's own norm, not a wholesale index shuffle, is already
  enough to destroy the entire degenerate spectrum. This matches X's
  already-measured 16-distinct-eigenvalue result (§12.1/§12.6) but sharpens
  it: the degeneracy isn't merely absent under X's *specific* shuffle
  construction, it is absent under *essentially any* generic direction,
  vanishingly close to the true tensor. Consistent with the degenerate
  locus being measure-zero / high-codimension in the space of bilinear
  maps (an 8-fold plus two 4-fold eigenvalue coincidence, holding
  simultaneously at *every* v in a 16-dimensional family, is an extremely
  restrictive condition by ordinary eigenvalue-perturbation-theory
  standards — satisfied by design for the CD-doubling construction, not by
  chance).
- **The one surviving direction.** Conjugating the true tensor by a random
  orthogonal transform `Q` (`T'(Qx,Qy) = Q·T(x,y)`, construction verified
  to 1.6×10⁻¹⁴) preserves the `(8,4,4)` spectrum **exactly** (eigenvalues of
  `L_v` vs. `L_{Qv}` match to 3.3×10⁻¹⁵) — as similarity-invariance
  requires. **This construction already exists in this project under
  another name: variant R**, orthogonally-conjugated K3, proven vacuous
  (`score_R(q,k;W) ≡ score_S(q,k;OW)` to 1e-10, `PHASE4_kernel_memo.md`) —
  absorbed by learned projections, not a behaviorally distinct model, and
  dropped from the grid for exactly that reason.

**Reading: CONFIRMED, within the construction methods actually available
to this project** (index/sign shuffles of the true tensor, the same family
X belongs to; orthogonal conjugation, the family R belongs to). Every
generic perturbation destroys the conditioning; the only perturbation
that preserves it is already a known non-control. No third direction was
found, tried, or is suggested by either check. This is **not** claimed as
a fully general classification theorem covering every conceivable bilinear
map on ℝ¹⁶ — that would require a much deeper rigidity proof this session
did not attempt — but as a targeted answer to the actual design question
("can this project build V3b"), the answer is no, and the failure is
principled rather than a lack of trying: the two directions this project's
own existing controls already explore (shuffle, orthogonal conjugation)
exhaust the extremes (destroy-everything vs. preserve-everything), with
nothing found in between.

**(b) Consequence: V3b is retired, not merely deferred.** `PHASE5_PLAN.md`
§4/§11 updated — V3b moves from "needs real design work" to "closed,
not-buildable by shuffle or conjugation; no other construction identified."
The conditioning-vs-algebra attribution question this control was meant to
answer is not resolved by this — it is reclassified as **not answerable by
a matched-control experiment at all**, only by the correlational evidence
in (c) below.

**(c) Free empirical companion (no training): does conditioning predict
the grid's extrapolation ordering at all?** `phase4_v3b_cond_regression.py`
computes a per-variant conditioning number for all six grid variants —
`median cond(L_v)` over 2000 random unit v for S (2.8106, theorem-fixed,
seed-independent) and X (41.75–42.96 across the three grid seeds); for the
four dense variants (D0p, D0, D1, Q0), which have no bilinear structure
tensor at all (`score(q,k) = qᵀk/√d_h` is literally the identity bilinear
form), **cond = 1.0 by explicit convention** (perfectly conditioned,
v-independent) — then regresses this against the real 18-run grid's own
`ppl@512`/`ppl@1024` (`p4_artifacts/*/summary.json`, no new training):

| | Pearson(cond, ppl) | Pearson(log cond, log ppl) | Spearman |
|---|---|---|---|
| ppl@512 | +0.429 | +0.409 | **−0.001** |
| ppl@1024 | +0.638 | +0.576 | **+0.263** |

**No clean relationship.** The moderate positive Pearson figures are driven
almost entirely by X being simultaneously the worst-conditioned point and
the worst-extrapolating point — a single high-leverage outlier, not a
population trend; the rank-based Spearman correlation (far less sensitive
to one outlier's exact magnitude) is essentially zero at ppl@512 and weak
at ppl@1024. **Sharper evidence: the four dense variants all share
cond=1.0 identically, yet their own ppl@1024 spans 18.68 (D0) to 47.18
(Q0) — a 2.5× range with zero conditioning variance to explain it.** S
(cond=2.81, worse-conditioned than every dense variant) lands at 34.90,
statistically indistinguishable from D1 (cond=1.0, 34.04) and squarely
inside the dense family's own spread — conditioning does not even cleanly
separate S from the dense pack, let alone explain X's separate deficit.
**Conclusion: conditioning is not a sufficient explanation for the grid's
extrapolation ordering** — whatever differentiates D0/D0p/D1/Q0 from each
other (positional scheme, MLP width, query-(in)dependence) accounts for
variance at least as large as anything attributable to conditioning, and
no new training run is needed to reach this conclusion (per the task's own
stated bar). This does not by itself rule out conditioning as *part* of
X's story specifically — S-vs-X is still a two-point comparison confounded
exactly as V3b was meant to unconfound — it rules out conditioning as a
*grid-wide* driver, which is the only claim this companion check can
support without the now-closed V3b control.

**Files**: `phase4_v3b_rigidity_check.py`, `phase4_v3b_cond_regression.py`
(both new, no training, reusable).

### 12.9 APM Stage A (2026-07-29/30): extrapolation-rung steering test —
mechanism-paper outcome

**Setup.** Per §12.7's locked pre-registration, `phase4_matched_N_trained.py
--rungs 512,1024` ran the validated correlation-matched-null instrument
(§12.5) against the same trained S/X checkpoints, on ctx=512 and ctx=1024
extrapolation-rung inputs instead of the ctx=256 val set — same
construction as `phase4_grid.py`'s own `length_gen_eval`, no new training.

**Sanity gate (checked before any trained number, as pre-registered):**
init-mode corr-matched median percentiles at ctx=512 were `[0.508, 0.480,
0.497]` (S) and `[0.450, 0.455, 0.497]` (X); at ctx=1024, `[0.525, 0.510,
0.545]` (S) and `[0.522, 0.505, 0.512]` (X) — all close to the 0.5
no-steering reference at both rungs, for both variants, every seed. The
correlation-matched null transfers cleanly to both extrapolation lengths;
the trained results below are trustworthy under it.

**A bug was caught before finalizing, not after: report it plainly.**
§12.7's own written formula, `z = (0.5 − mean_percentile) / (std/√n)`, has
the subtraction order backwards relative to the sign convention already
established and used throughout §12.5 (there, z is *negative* for a real
steering effect — e.g. "z≈−6.4" for trained S in-distribution, mean
percentile 0.41 < 0.5). Applying §12.7's literal formula to this section's
data produced *positive* z for the same "percentile below 0.5" condition,
which would make the pre-registered `z ≤ −3` HOLDS threshold
**structurally unsatisfiable regardless of the true effect** — a
self-inconsistency between the written formula and the written threshold,
not a new finding about the data. Caught by inspecting the first computed
table (all positive z where a real effect was expected) before writing
anything up. **Fixed to `z = (mean_percentile − 0.5) / (std/√n)`** — this
recovers the established SS12.5 sign convention exactly and is the
formula actually applied below. This is disclosed here in full rather than
silently corrected, because it is exactly the kind of after-the-fact
adjustment pre-registration exists to make visible: the **threshold
values, their meaning (HOLDS = a real steering effect at that magnitude),
and the decision rule's structure are unchanged** — only a subtraction-
order typo in translating that already-agreed meaning into a formula was
fixed, before any decision was read off it.

**Results** (pooled over 3 seeds × 200 points = 600, `phase4_stageA_decision.py`):

| variant | rung | pooled median pct | pooled z | per-seed median pct | call |
|---|---|---|---|---|---|
| S | 512 | 0.4275 | −4.43 | [0.355, 0.485, 0.435] (all <0.5) | **HOLDS** |
| S | 1024 | 0.4400 | −3.90 | [0.398, 0.438, 0.475] (all <0.5) | **HOLDS** |
| X | 512 | 0.4400 | −2.91 | [0.490, 0.438, 0.395] | AMBIGUOUS |
| X | 1024 | 0.4800 | −0.25 | [0.487, 0.453, 0.522] | COLLAPSES |

S's effect is not a resolution-floor artifact at either rung either
(`ratio_p5` — actual_min / null 5th-percentile — pooled medians ≈0.89–1.02
across seeds/rungs, well short of saturating near the trial-count floor,
same diagnostic as §12.5 item 4).

**Pre-registered decision:** S is HOLDS at **both** rungs, with per-seed
consistency satisfied at both (all 3 seeds individually below 0.5, not
just the pooled figure). X is AMBIGUOUS at 512 (borderline, `2≤|z|<3`,
defaults to COLLAPSES per §12.7) and COLLAPSES outright at 1024 (z=−0.25,
indistinguishable from no effect). **→ MECHANISM PAPER**: the clean
dissociation condition (S holds both rungs, X collapses/ambiguous both
rungs) is met, with no rung-split and no ambiguity requiring a tie-break
in S's favor — X's own numbers get *weaker* going from 512 to 1024, the
opposite of what an artifact-of-more-data account would predict if it
were simply "still deciding."

**What this does and does not establish.** Both S and X engage the K3
mechanism in-distribution (§12.5) — engagement alone does not explain the
S-vs-X extrapolation dissociation (§5), as already flagged. Stage A adds:
**only S's engagement transfers out of distribution; X's does not.** This
is a real, pre-registered, sign-bug-corrected-before-reveal result, not an
inference from end-state data the way Stage 0 item 1's trajectory reading
was (§1 of `PHASE5_stage0_findings_2026-07-26.md`) — it is measured
directly at the rungs in question. It does **not** by itself prove *why*
only S's engagement transfers — the kickoff document's own suggested
reading (S's structure is G₂-organized and coherent, per the bilaterality/
multiplicity-signature/annihilator-dimension discriminators already on
record — §8.4 / §12.1–§12.3 — while X's is an arbitrary bilinear map with
no such organizing structure, so its in-distribution engagement doesn't
generalize to unseen key distributions) is a plausible mechanism
connecting an already-established structural fact to a newly-measured
behavioral one, not something Stage A's numbers alone certify. That
connective argument, not a new measurement, is the paper's job now.

**Files**: `phase4_stageA_decision.py` (new), `p4_matched_N_trained_results_rungs_512_1024.npz`
(raw arrays, all 24 conditions), `p4_matched_N_trained_output_rungs.log`.
