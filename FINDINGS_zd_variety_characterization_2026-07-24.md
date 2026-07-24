# FINDINGS — X zero-divisor variety characterization (§4 of 2026-07-23 handoff)

**Date:** 2026-07-24
**By:** Claude (chat side) — analysis only, no repo changes proposed as edits
**Status:** class-level result; requires confirmation against the actual
`shuffled_structure_tensor` artifact (see §5 Caveats)
**Gate:** this is pre-verdict characterization, not grading. No H4a/H4c
language, no `RESULTS_phase4.md` conclusions.

---

## 1. Question

§4 asked: does the shuffled tensor X possess an accessible zero-divisor
variety? Branch points as pre-stated:

- infimum ≈ 0 → §3(e)'s non-descent is a real finding, positive survives
- infimum bounded away from 0 → §3(e) needs rewriting, S > X reduces
  toward "structured beats degenerate"

## 2. Method

Reduced the two-sided problem to a one-sided one. For fixed x, the map
y ↦ x ⊛ y is linear with matrix M_x[k,j] = Σ_i C[i,j,k] x_i, so

    min over unit y of ‖x ⊛ y‖  =  σ_min(M_x)

Therefore inf over unit x,y of ‖x ⊛ y‖ = inf over unit x of σ_min(M_x),
a 15-dimensional problem with the optimal y recovered exactly by SVD.
Minimized by L-BFGS on the sphere with analytic gradient
(dσ_min = uᵀ dM v), 150–400 random restarts per map.

Sedenion tensor built by Cayley–Dickson, **Baez convention**
(a,b)(c,d) = (ac − d̄b, da + bc̄), per standing orders.

**Method validation (all passed):**

| check | expected | measured |
|---|---|---|
| quaternions (division algebra) | infimum = 1 | 1.000000 |
| octonions (division algebra) | infimum = 1 | 1.000000 |
| (e₁+e₁₀)(e₅+e₁₄) in S | 0 | 0.000e+00 |
| dim of S's ZD variety | 14 (Moreno: ≅ G₂) | 14 |

The two division-algebra negative controls are the important ones: the
search does **not** return spurious zeros when none exist.

Shuffle spec `15 independent permutations + independent signs, one per k`
is ambiguous between two readings; **both were tested**:

- **Reading B** (literal, k-indexed): for each k ≥ 1, slice C[:,:,k] is
  an independent signed permutation matrix.
- **Reading A** (i-indexed): for each i ≥ 1, the map j ↦ k is an
  independent permutation with independent signs; e₀ left as identity.

12 independent draws per reading, plus a dense Gaussian bilinear map as
an "arbitrary map" reference.

## 3. Results

### 3.1 X has zero divisors — infimum is machine zero

| map | infimum σ_min |
|---|---|
| S (sedenion) | 8.8e-20 |
| X, reading B — 12 draws | max over draws 5.8e-19 |
| X, reading A — 12 draws | max over draws 1.2e-18 |

**24/24 draws, both readings: infimum ≈ 0.** This is the first branch.
§3(e) is not vacuous in the "nowhere to descend" sense — X had somewhere
to go.

### 3.2 The variety has the same dimension — and that dimension is generic

dim(ZD variety) inside S¹⁵ × S¹⁵ (ambient 30), from rank of the Jacobian
at converged solutions:

| map | dim | rank J |
|---|---|---|
| S sedenion | 14 | 16 |
| X_k draws 0–3 | 14 | 16 |
| X_i draws 0–3 | 14 | 16 |
| dense Gaussian map | 14 | 16 |

All 14. This is a pure dimension count: 30 ambient − 16 equations = 14.
**Possessing a 14-dimensional zero-divisor variety is generic for any
bilinear map R¹⁶ × R¹⁶ → R¹⁶.** It is not a property that distinguishes
the sedenions from an arbitrary map, and no claim should rest on it.

### 3.3 The reversal: X's variety is ~450× *more* accessible than S's

σ_min(M_x) at uniformly random unit x, 40 000 samples per map — i.e. how
close a random direction already is to a zero divisor, with no optimization:

| map | median | p05 | frac < 1e-1 | frac < 1e-2 | frac < 1e-3 |
|---|---|---|---|---|---|
| **S sedenion** | **0.478** | 0.230 | **0.0020** | **0.0000** | **0.0000** |
| X_k (5 draws) | 0.036–0.037 | 0.0033 | 0.920–0.929 | 0.146–0.150 | 0.0146–0.0155 |
| X_i (5 draws) | 0.039–0.040 | 0.0036 | 0.893–0.902 | 0.136–0.141 | 0.0134–0.0142 |
| dense Gaussian | 0.035 | 0.0032 | 0.941 | 0.155 | 0.0156 |

Ratio of near-ZD measure at threshold 1e-1: **X / S ≈ 440–465×**, uniform
across all 10 draws and both readings.

What is special about the sedenions is **not** that they have zero
divisors — arbitrary maps do too, of the same dimension. It is that
their zero divisors are *rare and hard to reach*: 0.2% of random
directions fall within 1e-1, versus ~92% for X, and **no** random
sedenion direction in 40 000 fell below 1e-2, versus ~15% for X.

## 4. Consequences

**(a) §4's first branch is taken.** X has zero divisors; §3(e) is not
vacuous for want of a target. The positive result is not killed by this
check.

**(b) A new confound appears in §3(e)'s statistic, not anticipated in the
handoff.** The fold-change metric is not baseline-matched. S starts far
from its variety (median 0.478); X starts nearly on top of its own
(median 0.037). A ~100× reduction from S's baseline lands ≈ 0.005; a
2.2× reduction from X's lands ≈ 0.017. **Most of the ~100× vs ~2.2×
gap is explainable as headroom rather than as differential
manifold-seeking.** The fold-change comparison should not carry weight;
the absolute floor should. This bears directly on A1 (magnitude
statistic).

**(c) A concrete discrepancy worth chasing.** §3(e) reports X *never*
places a pair below 1e-2. But ~15% of uniformly random directions in a
shuffled tensor are already below 1e-2 before any training. Under free
sampling of the sphere those two facts are hard to reconcile. Either:

  1. the ladder rotation / head-slot parameterization does not sample
     freely from S¹⁵ and is confining X away from its own variety — in
     which case §3(e) measures the parameterization, not the algebra; or
  2. min(r²) is normalized differently from σ_min here, and the two
     scales are not comparable; or
  3. the actual `shuffled_structure_tensor` differs from both readings
     modelled here.

All three are cheaply testable with the real artifacts. **This is the
next question, and it is sharper than the one §4 posed.**

**(d) Threat #2 (S > X loses attribution) is made worse, not better.**
S > X cannot be attributed to S offering a more accessible zero-divisor
manifold — S's is ~450× *less* accessible. If an algebra claim survives,
it has to rest on coherence/associator structure, not on zero-divisor
availability. Recommend the write-up not lean on ZD accessibility as the
mechanism.

**(e) §3(e) needs rewriting either way**, though not for the reason §4
anticipated. The correct statement is not "X had nowhere to descend"
(false) but something in the neighbourhood of "X began adjacent to its
variety and did not exploit it, while S began far from its own and
approached it" — and clause 2 of that is exactly what (b) and (c) put in
question.

## 5. Caveats — read before acting

1. **This is a class-level result, not a measurement of your artifact.**
   The actual `shuffled_structure_tensor`, the true structure tensor, and
   the ladder rotation were not available to this session. The result is
   uniform across 24 draws, two readings of the spec, and a dense
   Gaussian reference, and §3.2 shows the core of it follows from a
   dimension count — so it is very likely to hold for your specific
   draw. It still needs confirming against the real tensor.
2. **σ_min is a proxy for min(r²).** The mapping between the two has not
   been established. §4(c) may dissolve once it is.
3. Baez convention throughout. If the repo tensor uses a different CD
   convention, flag the mismatch rather than reconciling it here.
4. No autotopy/isotopy question is answered here. §3.3 says X's variety
   is more accessible; it says nothing about whether X is isotopic to S.

## 6. Suggested next actions (for Claude Code to verify, not applied here)

- **N1.** Re-run §3.1 and §3.3 on the *actual* seed-1337/1338 shuffled
  tensors. CPU-seconds. Confirms or breaks the class-level claim.
- **N2.** Resolve §4(c): sample min(r²) at random head-slot
  initialization under the real ladder rotation, for both S and X, and
  compare against the free-sphere numbers in §3.3. This separates
  "parameterization confines X" from "statistic scales differ."
- **N3.** Report absolute min(r²) floors and step-0 baselines alongside
  every fold-change already recorded, per §4(b). May be recoverable from
  existing logs without re-running anything.
- **N4.** Hold the A3-revised ablation design until N2 returns. §4 of the
  handoff wanted this answer before the ablation was designed; the answer
  moved the question rather than closing it.
