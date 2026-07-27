# PHASE 5 — PLAN

**Status:** DRAFT for Claude Code review. Nothing here is frozen.
Hypotheses in §5 are **proposals**, not pre-registrations — see §7 for the
freezing procedure, which now requires a power check that Phase 4 lacked.
**§2 corrected 2026-07-26** after repo-side verification: bilaterality is
Moreno (1997) Cor. 1.6, not new; S_sym withdrawn as a variant (§2.2, §4);
per-k shuffle row withdrawn (never implemented); pinned-seed numbers
replace distributional ranges. Full record: `PHASE5_verification_2026-07-26.md`.
**Date:** 2026-07-26
**Author:** Claude (claude.ai chat)
**Predecessor:** `RESULTS_phase4.md`, grid 18/18, verdicts read out.

---

## 1. Background — where Phase 4 landed

**Verdicts.** H4a NEGATIVE (S trails the flops-matched dense baseline by
0.1600 against a 0.0076 margin — a factor of 21). H4b′ not evaluated.
H4c PASS (six heads cross the frozen bar in all three seeds; five in layer 0).
Wall-clock bound EXCEEDED at 2.39× D0p.

**One-line summary:** *the mechanism engages and does not pay.*

**Primary positive.** The S–X double dissociation, 3/3 seeds with complete
separation on all three axes — X fits the training distribution better, S
extrapolates better at both rungs. S and X are identical in d_model, mlp
width, parameter count to the digit, flops/token, and wall-clock to within 6
seconds; the structure tensor is the only difference. This is the only
comparison in the grid with a single differing variable, and it is what
survived every challenge raised against it.

**What Phase 4 could not establish, and why it matters.**

1. **Attribution.** X differs from S in at least five ways simultaneously —
   non-unital, non-flexible, non-power-associative, ~16× worse conditioned,
   and different null-set accessibility. The dissociation attributes the trade
   to *the structure tensor*, not to zero-divisor structure specifically. No
   valid-algebra-wrong-frame control (the V3 analogue) exists in the design.
2. **Absolute standing.** Both tensor variants lose to every dense baseline.
   Normalized by each variant's own in-distribution perplexity, S degrades
   5.78× from ctx 256 to 1024 — third of six, behind D0 (4.01×) and D0p
   (5.48×). S's extrapolation advantage holds only over X and over D1.
3. **The width confound.** D1 achieves flops parity with S by widening the MLP
   to 1992, and rung performance sorts monotonically by MLP width across
   D0 (1536) → S (1824) → D1 (1992). No dense variant was run at S's own
   mlp = 1824, so "S beats flops-matched dense on extrapolation" is not
   separable from "wide MLPs extrapolate badly."
4. **Can't vs doesn't.** Whether X *can* reach low r² under the real
   parameterization was never measured. Two seeds converging near 0.033
   suggested a floor; seed 1339's 0.02818 broke that reading.

---

## 2. Bilateral asymmetry — Moreno 1997, repo-verified 2026-07-26

**Verified by Claude Code against the repo's own tensors**
(`PHASE5_verification_2026-07-26.md` has the full record; summary below).
**Not a new result.** Bilaterality of Cayley-Dickson zero divisors for
n ≥ 4 is Corollary 1.6 of G. Moreno, "The zero divisors of the
Cayley-Dickson algebras over the real numbers," arXiv:q-alg/9710013 (1997)
— already cited by Reggiani, already in this project's citation chain. The
chat-side derivation (conjugation is an anti-automorphism + zero divisors
are purely imaginary ⇒ x⊛y=0 ⇒ y⊛x=0) is a correct one-line re-derivation
of that corollary, not an independent finding. Framed here as **verified
against Moreno 1997**, not as new.

Zero-divisor annihilation in the sedenions is **two-sided**. In the
repo's shuffled tensor (X, the per-i construction —
`phase4_layers.shuffled_structure_tensor`; no other shuffle variant exists
in this codebase) it is not.

| | ordered ZD pairs | reverse also zero | bilateral |
|---|---|---|---|
| **S** | 336 | 336 | **100%** |
| X, seed 1337 | 96 | 2 | 2.1% |
| X, seed 1338 | 92 | 0 | 0.0% |
| X, seed 1339 | 66 | 0 | 0.0% |

(Pinned-seed values from the repo's own construction, replacing the
original draft's non-pinned distributional range.)

On general unit vectors — locate x on the ZD variety by σ_min (inner
minimization over y is exact via SVD), then measure ‖y⊛x‖ against
‖x⊛y‖:

- **S**: forward and reverse medians agree to 3 significant figures
  (4.54×10⁻¹¹ vs. 4.539×10⁻¹¹) — bilateral at general points, not just at
  basis two-blades. Numerics stop at ~1e-11 (optimizer convergence, not a
  live mathematical question — see below).
- **X**: forward medians ~7×10⁻¹² (genuinely on the one-sided variety),
  reverse medians 0.96–1.03 — indistinguishable from the random-pair
  baseline (mean 0.998).

**Preferred certificate, added 2026-07-26:** Moreno's Cor. 1.5 gives a
stronger, cheaper check that holds on *all* unit pairs, not just the
14-dimensional zero locus: ‖xy‖=‖yx‖ everywhere. Checked on 5000 random
unit pairs per tensor: S is symmetric to float64 machine precision
(max diff 4.4×10⁻¹⁶, 0.00% median relative asymmetry, no exceptions); X's
median relative asymmetry sits at 15.2–15.6% regardless of seed. One
`einsum` and a norm comparison, no need to locate the zero-divisor variety
at all. This is now the recommended X-inequivalence certificate,
superseding the Reggiani-isometry framing proposed in
`PRIOR_ART_REVIEW_zda.md` §3.1(a) — that framing (null-set smoothness,
`phase4_reggiani_certificate.py`) remains valid on its own terms but this
identity is cheaper and holds everywhere.

Note X frequently possesses *more* one-sided zero divisors than S and
essentially no bilateral ones.

### 2.1 Why this matters

This is a **binary structural invariant** (Moreno 1997, not this project's
finding), unlike the two candidate discriminators used so far:

| candidate invariant | S | X | discriminates? |
|---|---|---|---|
| dim of ZD variety | 14 | 14 | **no** — generic to any bilinear map |
| r² descent magnitude | 78–121× | 1.0–2.2× | yes, but continuous and contestable |
| **bilaterality (Moreno Cor. 1.6)** | **100%** | **0%** | **yes, binary** |
| **norm symmetry (Moreno Cor. 1.5)** | **exact, everywhere** | **~15% median asymmetry** | **yes, binary, and cheaper** |

Both bottom rows follow the same theorem; the norm-symmetry row is the
operative certificate going forward (§2 above).

### 2.2 S_sym is not a variant — it is S, by algebraic identity

Because ‖P⊛Q‖ = ‖Q⊛P‖ **identically** for S (not just on the zero set —
confirmed everywhere to 4.4×10⁻¹⁶), the symmetrized score

  ‖P⊛Q‖ + ‖Q⊛P‖ = 2‖P⊛Q‖

is an exact constant multiple of S's own score, not a new function of
(P,Q). γ is a learned per-head scalar multiplying the score (init 1.0,
spec §2), so training "S_sym" would measure γ-initialization sensitivity
(effectively γ init 2.0), not algebra. **S_sym is withdrawn as a variant**
(removed from §4's table; see §5 and §8 for the corresponding narrowing of
H5a and the compute budget). This was originally stated as a prediction
("S_sym ≈ S") to be tested; it is not testable in that sense — it is true
by definition, derivable without running anything.

**X_sym is unaffected and remains in the design.** X's two conditions are
independent (§2's ~15% median asymmetry, ~1.0 median reverse-norm at
general points), so symmetrizing X is a real, non-redundant test: whether
two independent near-zero conditions are jointly satisfiable in an
arbitrary bilinear map (they should not be, generically).

---

## 3. Goals

In priority order.

**G1 — Attribution.** Determine whether the S–X trade is caused by algebraic
coherence (bilateral annihilation) or by any of the confounded properties X
also lacks. This is the single largest limit on what the paper can claim.

**G2 — Falsification of the sparsity alternative.** ZDA produces structured
sparse attention. So does entmax, with no algebra whatsoever. If entmax
reproduces the extrapolation trade, the finding is about sparsity and the
algebra is incidental. This must be ruled out before any algebraic claim is
published.

**G3 — Close the width confound.** Run dense at mlp = 1824. Until this exists,
no S-vs-dense extrapolation comparison is interpretable (§1.3).

**G4 — Cost.** Five of six H4c-crossing heads are in layer 0. Test whether a
layer-0-only hybrid retains the extrapolation character at a fraction of the
2.39× wall-clock penalty. This is the only route by which the architecture
becomes practically relevant rather than merely interesting.

**Non-goal.** Beating dense baselines. Phase 4 established S does not, and
Phase 5 is not designed to change that. Any regime hunt is a separate phase
(§11).

---

## 4. Variants

New:

| variant | construction | serves |
|---|---|---|
| **X_sym** | shuffled tensor, symmetrized score ‖P⊛Q‖+‖Q⊛P‖ | G1 |
| **E0** | dense attention + α-entmax (no algebra) | G2 |
| **D1824** | dense, mlp = 1824, everything else as D0p | G3 |
| **S_L0** | S at layer 0, dense at layers 1–5 | G4 |

**S_sym dropped, 2026-07-26** (see §2.2): ‖P⊛Q‖+‖Q⊛P‖ = 2‖P⊛Q‖ identically
for S, since the two norms are equal everywhere (Moreno Cor. 1.5, verified
to 4.4×10⁻¹⁶). Training it would only measure γ-init sensitivity.

Carried from Phase 4 as reference at 3 seeds each, **not re-run**: S, X, D0p,
D0, D1, Q0.

Seeds 1337, 1338, 1339 throughout, for comparability.

---

## 5. Candidate hypotheses (PROPOSALS — not frozen)

Each requires a power check (§7.2) before it may be frozen.

**H5a — attribution (restated 2026-07-26, narrower than the original
draft).** The S_sym clause is withdrawn — "S_sym matches S" is true by
algebraic identity (§2.2), not an empirical test, and cannot contribute
evidence either way. What remains: **X_sym falls toward the Q0 floor
rather than tracking X.** This alone is now the whole of H5a: whether
X's two independent near-zero conditions (§2's ~15% median asymmetry) are
jointly unsatisfiable under training, the way they are under free
sampling. **This is a real loss of power, not just a reframing** — the
original design used S_sym as an internal control (a variant that
*should* look unchanged) to validate that the symmetrized-score
construction itself wasn't introducing some confound; without it, a
positive X_sym→Q0 result is consistent with the attribution claim but no
longer cross-checked against a variant expected to be neutral. Flagged,
not fixed — no substitute internal control identified yet.

**H5b — sparsity falsification.** E0 does *not* reproduce the S-over-X
extrapolation ordering on normalized degradation. If E0 does reproduce it, G2
fails and the algebraic framing must be withdrawn or heavily qualified.

**H5c — width confound.** D1824's normalized extrapolation (ppl@1024 / ppl@256)
differs from S's by more than the dense-family seed spread. Note the dense
family's seed spread is large — D0p's ppl@512 swings 46% across seeds — which
sets a demanding floor on what margin is readable here.

**H5d — locality and cost.** S_L0 retains S's normalized extrapolation within
the S seed-spread at wall-clock below 1.5× D0p.

---

## 6. Staging

Gated. Each stage completes and reads out before the next is designed.

**Stage 0 — zero compute, from existing artifacts. Do first.**

- ~~Full per-eval r²_min trajectories for S and X, all 3 seeds, all 13 eval
  points. Tests the **early-lock** hypothesis: does X flatten after the first
  one or two evals while S continues descending?~~ **DONE 2026-07-26** —
  `PHASE5_stage0_findings_2026-07-26.md` §1. Confirmed in a sharper form:
  X makes one small move away from init by the first eval then shows a
  slope statistically indistinguishable from flat for the rest of training
  (mean log-slope +0.015/eval); S shows sustained negative log-slope
  through nearly the whole run (mean −0.052/eval), with min-ever landing
  late in training for 2 of 3 seeds, not concentrated early.
- ~~cond(L_x) evaluated at the *real* init q,k distribution (reuse the N2
  simulator), S and X. Confirms or dissolves the conditioning objection
  quantitatively.~~ **DONE 2026-07-26** —
  `PHASE5_stage0_findings_2026-07-26.md` §2. Confirms: real-init medians
  (q and k, all 3 X seeds) match the free-sphere prediction to ~1%
  (S≈2.8, X≈43–44) — the ~15–16× conditioning gap is not a sampling
  artifact, it's what the model's actual init vectors experience.
- Constrained-infimum check: freeze the tensor, disable the LM objective,
  optimize q,k projections directly against r². Returns each variant's
  reachable floor and settles can't-vs-doesn't. **BLOCKED 2026-07-26** —
  requires the trained grid checkpoints (~3GB, Drive-only per §9); confirmed
  none exist locally. Not substituted with a fresh-init version without
  approval, since that would answer a different question (see
  `PHASE5_stage0_findings_2026-07-26.md` §3 for why).
- ~~Verify §2's bilaterality result against the repo's own tensors.~~ **DONE
  2026-07-26** — `PHASE5_verification_2026-07-26.md`. Also surfaced that
  bilaterality is Moreno (1997) Cor. 1.6, not new, and identified a cheaper
  everywhere-holding certificate (Moreno Cor. 1.5, norm symmetry) that
  supersedes the original bilaterality-at-zero framing; dropped S_sym as a
  variant (§2.2, §4).

Stage 0 may reshape everything below it. It costs an evening and no units.
**Reshape check, 2026-07-26: neither completed item changed the framing** —
both held up rather than dissolved (see findings doc "Bottom line"). Stage 1
is not blocked on item 3; nothing in Stage 1/2 depends on the
constrained-infimum result specifically.

**Stage 1 — cheap training runs.** D1824 and E0. Both are dense-class
(~0.6–0.7 h/run), ~4 h total for 6 runs. Closes G2 and G3, either of which can
invalidate the framing before expensive runs begin.

**Stage 2 — expensive training runs.** X_sym, S_L0 (S_sym dropped 2026-07-26,
§2.2). X_sym computes two tensor products and should be budgeted at up to 2×
S's 1.44 h.

**Gate between stages 1 and 2:** if E0 reproduces the extrapolation trade
(H5b fails), stop and re-scope. Spending Stage 2's budget on an algebraic
attribution question is only justified if the algebra is still in play.

---

## 7. Pre-registration discipline

### 7.1 Carried from Phase 4 (these worked)

- Completeness gate: no grading, no verdict language, no results-document
  conclusions until the grid is complete and the summary reads itself out.
- Amendments frozen in the repo before the data they concern exists;
  otherwise labeled post-hoc. Under any uncertainty, take the weaker label.
- Negative results receive the same care as positive ones.
- Baez convention. Flag convention mismatches; never average them.

### 7.2 New requirement — the power check

**Phase 4's H4c tested reliability, not magnitude.** Its bar scaled to each
head's own step-0 variance, so it detected *consistent* movement rather than
*large* movement — and X crossed the same bar in two heads. The criterion
therefore could not discriminate the variant it was designed to characterize
from its control, and the magnitude statistic had to be added post-hoc (A1).

**Requirement:** before any Phase 5 criterion is frozen, demonstrate on
existing Phase 4 data or on simulation that it separates the hypotheses it is
meant to separate. A criterion that cannot be shown to discriminate is not
frozen.

### 7.3 New requirement — readout engineering

From the H4c `nan` incident (`RESULTS_phase4.md` §7.1):

- **A verdict must never be emitted from a non-finite statistic.** Raise, do
  not print, on any non-finite margin. (Implemented at 415506e; keep.)
- **No parameter may be shared between an execution loop and a readout loop.**
  Grading code discovers its inputs from disk; it never trusts a CLI argument
  that also controls training.
- Grading refuses to run on partial seeds rather than degrading silently.

---

## 8. Compute budget

Estimated at the measured rate of **≤11.2 units/hr on A100** (verified
2026-07-24, superseding the earlier ~13/hr estimate).

| stage | runs | est. hours | est. units |
|---|---|---|---|
| Stage 0 | 0 | — | 0 |
| Stage 1 (D1824, E0) | 6 | ~4 | ~45 |
| Stage 2 (X_sym, S_L0) | 6 | ~13 | ~145 |

**Stage 2 revised 2026-07-26**: down from 9 runs to 6 (S_sym dropped, §2.2/§4).
The original ~19h/~215-unit estimate was given for all three variants
combined without a per-variant breakdown; the table above removes X_sym and
S_L0's roughly two-thirds share rather than re-deriving hours from scratch,
since S_sym and X_sym were budgeted similarly (both "up to 2× S's 1.44h" per
§6) — treat ~13h/~145 units as an approximate, not re-measured, estimate.

Stage 2 still substantially exceeds any remaining balance and requires a
purchase decision. **This is a reason to take the staging seriously rather
than a formality:** Stage 0 is free and may reduce Stage 2's scope; Stage 1
is cheap and can invalidate Stage 2 entirely.

Keep a reserve. The Phase 4 grid was interrupted once mid-run by exhaustion,
and the recovery cost a clean re-run.

---

## 9. Engineering notes

- `phase4_grid.py` is frozen at sha256 `65d419e05b6ec74d…` as the script that
  produced the 18-run grid; readout corrected at 415506e. Phase 5 should be a
  **new script**, not an edit, so Phase 4's provenance stays intact.
- Bit-exact reproducibility was demonstrated across sessions, runtimes, and
  physical A100 instances (`RESULTS_phase4.md` §6.3). Preserve whatever makes
  that true — TF32 disabled, cached tokenizer, `meta.json` early-return.
- Analysis artifacts now live in the repo at `p4_artifacts/` (76 files,
  0.3 MB). Checkpoints remain Drive-only at ~3 GB and are required for the
  Stage 0 constrained-infimum check.

---

## 10. Inherited open items

| item | stage | notes |
|---|---|---|
| Constrained-infimum (can't vs doesn't) | 0 | **BLOCKED 2026-07-26** — needs checkpoints, none found locally |
| ~~cond(L_x) at real init~~ | 0 | **DONE 2026-07-26**, `PHASE5_stage0_findings_2026-07-26.md` §2 |
| ~~Early-lock trajectory analysis~~ | 0 | **DONE 2026-07-26**, owner's hypothesis, confirmed in sharper form, §1 |
| ~~Verify bilaterality vs repo tensors~~ | 0 | **DONE 2026-07-26**, §2, `PHASE5_verification_2026-07-26.md` |
| ~~Verify 84/336 vs repo enumerator~~ | 0 | **DONE 2026-07-26**, prior art §8.4.8 now RESOLVED |
| A3-revised γ dose-response | 1 | inference-only, plausibly free tier |
| S descent range correction (77.7–121.3×) | — | amendment text fix |

---

## 11. Explicitly out of scope

Closed unless deliberately reopened:

- **Regime hunting** (modular arithmetic, algorithmic tasks, equivalence-class
  retrieval). Interesting and mechanism-derived, but speculative and a
  separate phase. Phase 5 is attribution, not application.
- **The full V3 ladder** (isomorph / sign-flip / alternate-algebra controls).
  The symmetrized test may answer what V3b was for at a fraction of the cost.
  Revisit only if H5a is inconclusive.
- **Beating dense baselines.** See §3, Non-goal.
- Carried closed doors from Phase 4: no K1, no variant R, no β-equivalence
  test for K3, `ZD_PAIR` unchanged, γ/β never weight-decayed.

**Reopened:** entmax, previously non-graded. That standing order was
conditional on the Phase 4 completeness gate, which has lifted. Entmax enters
Phase 5 as a graded control (E0), not as a variant of interest.
