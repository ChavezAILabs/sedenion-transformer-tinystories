# BCDI EIGENTHEORY — FINDINGS, 2026-07-27

**Source:** Biss, Christensen, Dugger, Isaksen, *Eigentheory of Cayley–Dickson
algebras*, Forum Math. 21 (2009) 833–851; arXiv:0905.2987. Full text read.
**From:** chat side. **Status:** proposal for Claude Code verification.
**Companion to:** `PRIOR_ART_addendum_2026-07-27.md` (rev. B), which flagged
this paper as priority for §2.7.

**Reading convention.** Quoted results are labelled with the paper's own
numbering. Anything marked **[derived]** is chat-side reasoning and must be
independently re-derived before it is relied on — see §6.

---

## 0. Summary

The spectrum of sedenion left multiplication is completely known and has
**three eigenvalues at multiplicities 8, 4, 4**. Consequences:

1. cond(L_v) has a **closed form**. HANDOFF §2.7 Item 2 becomes analytic.
2. The **multiplicity signature supersedes** the rank-deficiency statistic
   proposed in the companion addendum §4 — same discrimination, available at
   every point rather than only at variety points, no variety sampling.
3. **Stage 0 item 3 is closed for S in closed form.** The global infimum of r²
   over Q, given P, is an explicit function of P. It remains open for X.
4. Koebisu's Theorem 3.7 **appears to be a corollary** of this 2009 paper,
   which he does not cite. [derived]

---

## 1. The setup, as the paper states it

**Definition 3.1.** M_a := (1/|a|²) L_{a*} L_a. The *eigenvalues of a* are the
eigenvalues of M_a.

Since L_{a*} is the adjoint of L_a (**Lemma 2.3**), M_a is symmetric PSD, and
its eigenvalues are the **squared singular values of L_a divided by |a|²**.
This is the bridge to the repo's cond(L_x) measurement.

**Proposition 3.9.** M_a is diagonalizable with non-negative eigenvalues;
distinct eigenspaces are orthogonal.

**Lemma 3.8.** ker M_a = ker L_a. Hence a is a zero divisor iff 0 is an
eigenvalue.

**Proposition 3.17.** The eigenvalues of a sum to 2ⁿ (with multiplicity).

**Proposition 3.20.** Every eigenspace has real dimension a multiple of 4.
*This is the structural reason behind BDI's multiple-of-4 annihilator bound.*

---

## 2. The sedenion spectrum

Write v = u + w·e₈ with u, w ∈ Im(𝕆), θ the angle between them. (BCDI's C₄ =
span{e₀, e₈}, so C₄^⊥ is exactly Koebisu's x₁ = x₂ = 0 locus.)

**Corollary 7.3.** For u, w imaginary and linearly independent, the eigenvalues
of (u,w) are

| eigenvalue | multiplicity |
|---|---|
| 1 | 8 |
| 1 + 2‖u‖‖w‖sinθ / (‖u‖²+‖w‖²) | 4 |
| 1 − 2‖u‖‖w‖sinθ / (‖u‖²+‖w‖²) | 4 |

**Proposition 3.10** extends this to general v (with real parts): eigenvalues
become sin²φ + λcos²φ, where cos²φ is the fraction of ‖v‖² lying in C₄^⊥.

**Proposition 7.1 / Corollary 7.3 remark.** If u, w are linearly dependent,
sinθ = 0 and 1 is the only eigenvalue — the element is alternative.

### 2.1 Consolidated form [derived]

Set

> **S(v) := 2·√(‖u‖²‖w‖² − ⟨u,w⟩²) / ‖v‖²**,  S ∈ [0,1]

using ‖u‖‖w‖sinθ = √(Gram(u,w)). Then the eigenvalues of any nonzero v ∈ 𝕊 are

> **1 (mult. 8), 1+S (mult. 4), 1−S (mult. 4)**

and therefore

> **cond(L_v) = √( (1+S) / (1−S) )**

The real parts x₁, x₂ enter only through ‖v‖² in the denominator of S.

---

## 3. Cross-checks [derived]

Two independent consistency checks, both exact.

**(a) Against Koebisu's determinant.** Koebisu's
D₂ = ‖v‖⁴ − 4(‖u‖²‖w‖² − ⟨u,w⟩²) = ‖v‖⁴(1 − S²), so

  det L_v = D₁⁴D₂² = ‖v‖⁸ · ‖v‖⁸(1−S²)² = ‖v‖¹⁶(1−S²)².

From the spectrum, det M_v = ∏λᵢ = 1⁸·(1+S)⁴·(1−S)⁴ = (1−S²)⁴, and
det M_v = (det L_v)²/‖v‖³², giving det L_v = ±‖v‖¹⁶(1−S²)². **Exact match.**

**Consequence.** Koebisu's Theorem 3.7 is the product of BCDI's Corollary 7.3
eigenvalues with multiplicities. He cites BDI I [5] and Reggiani [6] but not
this paper. This is an observation about logical dependence, not about
conduct — his G₂-reduction-plus-block-computation route is genuinely
different, and the determinant-level framing is his own contribution. **Do not
write this up as a priority claim.**

**(b) Against the zero-divisor case.** At a zero divisor, ‖u‖ = ‖w‖ and
⟨u,w⟩ = 0 (and x₁ = x₂ = 0), so S = 1 and the spectrum collapses to
**0, 1, 2 at multiplicities 4, 8, 4**. That is **Proposition 7.4** verbatim,
which further gives

  Eig₀(u,w) = {(x, −uw·x/‖uw‖) : x ∈ ⟨⟨u,w⟩⟩^⊥}, dimension 4.

**Third independent confirmation of annihilator dimension 4** — after BDI's
bound and the Koebisu/Reggiani geometry — and the first that writes the
eigenspace down rather than bounding it.

---

## 4. Revision to the companion addendum §4

**The rank-deficiency statistic is superseded.** Replace with:

> **Eigenvalue multiplicity signature of M_P.**
>
> - **S: exactly three distinct eigenvalues at multiplicities (8, 4, 4).**
>   Theorem-fixed (Cor. 7.3), at *every* nonzero P, not only on the variety.
> - **X:** 16 distinct singular values expected, if the shuffle behaves like a
>   generic bilinear map. Unknown — this is the test.

Why this is strictly better than rank deficiency:

- **No variety sampling.** Rank deficiency is only defined at variety points
  and requires finding them first. The signature is defined everywhere.
- **Higher information.** A multiplicity partition of 16 rather than one
  integer.
- **Subsumes §2.7 Item 2.** cond(L_v) falls out of the same eigendecomposition,
  so the conditioning measurement and the structure test become one
  computation.
- **Meets §7.2 without a pilot**, for the same reason as before: the S value
  comes from a theorem, so separation is arguable in advance.

Cost: one symmetric eigendecomposition of a 16×16 matrix per sampled P.

---

## 5. Stage 0 item 3 — closed for S, open for X

**Proposition 4.4.** For all x, √(λ⁻)·|a||x| ≤ |ax| ≤ √(λ⁺)·|a||x|, with
equality on the left iff x ∈ Eig_{λ⁻}(a) (and correspondingly on the right).

Applied to the ZDA score r² = ‖P⊛Q‖², with λ⁻ = 1 − S(P): [derived]

> **min over unit Q of r²  =  (1 − S(P))·‖P‖²**, attained exactly on the
> 4-dimensional subspace Eig_{1−S}(P).

So on the S side the global infimum question of HANDOFF §3.1 **does not need a
restart search**. The infimum is a closed-form function of P alone, and the
attaining set is an explicit 4-dimensional subspace.

**This does not transfer to X**, which has no corresponding theorem. The
asymmetry is itself the finding: the "can't vs doesn't" question is only
empirically open on one side, and the §3.1 search should be re-scoped to X
alone.

**Note for §3.1's framing.** The infimum above is over *free* unit Q. The
reachability caveat raised earlier still stands — q, k are outputs of learned
projections applied to real activations, and Eig_{1−S}(P) being 4-dimensional
out of 16 says nothing about whether the projections can land in it. The
closed form removes the free-vector question, not the reachability question.

---

## 6. What must be re-derived before use

Everything in §2.1, §3, and §5 marked **[derived]** is chat-side algebra from
quoted results, not quoted itself. Specifically:

1. **S(v) = 2√(Gram)/‖v‖² and cond = √((1+S)/(1−S)).** Re-derive from Cor. 7.3
   + Prop. 3.10. Then check numerically against the existing §2.7 cond(L_x)
   samples for S — it should reproduce the measured median to floating-point,
   not approximately. **If it does not, §2 through §5 all fail and should be
   discarded.** This is the single check that gates everything else.
2. **D₂ = ‖v‖⁴(1−S²) and the det cross-check in §3(a).**
3. **r²_min = (1−S(P))‖P‖² in §5.**

Item 1 is cheap and decisive. Run it first.

---

## 7. Smaller items for the review

- **Lemma 2.10:** ‖xy‖ = ‖yx‖ for all x, y in any Aₙ, proved in six lines from
  Lemma 2.4, with **no dimension restriction**. The review's §3.1(a) addendum
  credits Moreno 1997 Cor. 1.5 with the qualifier "dimension ≥ 16." BCDI give
  a second, self-contained proof without that qualifier. Add as a co-citation
  for the X-inequivalence certificate.
- **Remark 3.12:** zero divisors are always orthogonal to Cₙ — i.e. Koebisu's
  x₁ = x₂ = 0 — cited to [M1, Cor. 1.9] and [DDD, Lem. 9.5]. Note HANDOFF §2.6
  attributes "doubly pure" to Moreno **1997** Cor. 1.9 (q-alg/9710013) while
  BCDI's [M1] is Moreno **1998** (Bol. Soc. Mat. Mex.). Possibly two different
  papers with coinciding numbering. **Flag, do not reconcile** — standing rule.
- **§9 open questions.** Question 9.4 asks for the space of possible spectra of
  zero-divisors in Aₙ; Question 9.5 asks for the characteristic polynomial of
  elements of Aₙ. Koebisu 2025 bears on 9.5 at n = 4 (the determinant, not the
  full characteristic polynomial). Worth one line in the review as context for
  why the Koebisu paper exists.
- **Prop. 4.7 / Cor. 4.8:** all eigenvalues of Aₙ lie in [0, 2ⁿ⁻³], sharp. At
  n = 4 that is [0,2], consistent with §3(b).
