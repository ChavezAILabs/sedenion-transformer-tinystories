# PRIOR ART ADDENDUM — 2026-07-27 (rev. B)

**From:** chat side. **Status:** proposal for Claude Code verification.
**Supersedes rev. A** of the same date, which asserted a missing-citation gap
that does not exist. See §5.
**Diffed against:** `PRIOR_ART_REVIEW_zda.md` as of 2026-07-27 upload.

---

## 1. RETRACTION — HANDOFF §3.3's premise is false

HANDOFF 2026-07-26 §3.3 lists Koebisu (arXiv:2512.13002) and
Biss–Dugger–Isaksen under the heading **"New prior art to check against the
review."** Neither is new. Both are already in `PRIOR_ART_REVIEW_zda.md`:

| item | location | since |
|---|---|---|
| Biss–Dugger–Isaksen, *Large annihilators* (2008) | §3 supporting literature, line 188 | 2026-07-20 pass |
| Koebisu, arXiv:2512.13002 | §3 supporting literature, lines 190–191 | 2026-07-20 pass |
| Koebisu, again | §8.3, line 475, as "the one Reggiani citation was already known" | 2026-07-23 |

BDI is moreover already put to work in §3.1(b), named as "the direct reference
for how annihilator dimension scales with doubling."

**Rev. A of this addendum repeated the error**, asserting in its §4 that the
review "was missing the Biss–Dugger–Isaksen–Christensen cluster as a block."
It was not. This is HANDOFF §7 calibration item 1 recurring — extrapolating
from a handoff summary without reading the source document — committed in the
same session in which it was named.

**Genuinely absent from the review:** only

- Biss–Christensen–Dugger–Isaksen, *Large annihilators II*, Bol. Soc. Mat.
  Mex. 13 (2007) 269–292; arXiv:math/0702075. Not read.
- Biss–Christensen–Dugger–Isaksen, *Eigentheory of Cayley–Dickson algebras*,
  Forum Math. 21 (2009) 833–851; arXiv:0905.2987. Not read. **Priority** — an
  eigentheory of CD multiplication operators is upstream of the cond(L_x)
  measurement in HANDOFF §2.7.

---

## 2. CORRECTION — HANDOFF §3.4 is Moreno 1998 / Koebisu Cor. 3.8

§3.4 records, as a speculative G4 lead:

> The ZD locus has an exact closed form, verified to machine precision on 40
> variety points: Re(v₁) = Re(v₂) = 0, ‖u‖ = ‖w‖, ⟨u,w⟩ = 0.

This is the classical characterization of nonzero sedenion left zero-divisors.
Koebisu states it in his introduction in these variables and attributes it to
Moreno [3] (Bol. Soc. Mat. Mex. 1998); it recurs as his **Corollary 3.8**,
there *derived* from the determinant factorization rather than assumed.

**The failure is not "did not search."** Koebisu has been cited in our own §3
since 2026-07-20. The review's entry for it is one sentence at abstract level
— determinant of left multiplication, algebraic locus singular, normalized
locus smooth — and the paper was never read past that. A result inside a paper
in our bibliography was then rediscovered numerically on 40 sample points.

**Actions:**

1. §3.4 cites Moreno 1998 and Koebisu Cor. 3.8. The 40-point check is
   relabeled a kernel consistency check, not a finding.
2. Review §3's Koebisu entry is expanded from abstract-level to content-level
   (see §4 below for the content).
3. See §5 for the systemic version of this.

### 2.1 §3.4's caveat undersells the lead

§3.4 notes the closed form tests a *single* element while r² = ‖P⊛Q‖² is
pairwise, so it is "a cheap test for 'does P have a good partner,' not 'is Q
that partner.'" True but weak. By BDI every sedenion zero divisor has an
annihilator of dimension **exactly 4** — no exceptional strata. So the P-test
is complete and uniform, and the partner set is a full 4-dimensional subspace
recoverable as ker M(P). For G4 that is a constructive two-step, not a screen.

---

## 3. The dimension-14 measurement decomposes — and the total is a coincidence

### 3.1 HANDOFF §2.6's stated reason is wrong

§2.6 dismisses variety dimension as a discriminator: "14 for both — generic to
any bilinear map." Per review §3.1(a), the 14 is **dim G₂** (Reggiani).
Separately, a generic bilinear map on ℝ¹⁶ gives normalized 30 − 16 = 14.

Two different 14s that coincide numerically. **§2.6's conclusion survives; its
reason does not** — and the wrong reason is what made the decomposition look
uninteresting.

### 3.2 The decomposition, forced three ways

Review §3 defines 𝒵(𝕊) ⊂ 𝕊 × 𝕊 as the set of **normalized** pairs with zero
product, so all figures below are normalized.

- **Reggiani:** 𝒵(𝕊) ≅ G₂, dim 14.
- **Koebisu Thm 4.2:** the image of the projection to the first factor is
  V₂(ℝ⁷), dim 2·7 − 3 = 11.
- **Fiber:** V₂(ℝ⁷) ≅ G₂/SU(2). G₂ acts transitively on S⁶ with stabilizer
  SU(3); SU(3) acts transitively on S⁵ with stabilizer SU(2) — the exact
  transitivity chain Koebisu's Lemma 3.6 runs for a different purpose. So the
  fiber of 𝒵(𝕊) → V₂(ℝ⁷) is SU(2) ≅ S³, **dim 3**.
- **BDI, independently:** annihilator dimension ≤ 2ⁿ − 4n + 4, every multiple
  of 4 up to the bound realized. At n = 4 the bound is 4 and the permitted
  values are 0 and 4, so every zero divisor has dim Ann = 4 — **3 normalized.**

11 + 3 = 14. Three sources agree on a number none of them states jointly.

Note this is a *different* submersion from Reggiani's own (over the symmetric
space of quaternion subalgebras of 𝕆, fibers S³ × S³, 8 + 6 = 14). Both are
valid fibrations of G₂. **No new structure is claimed here** — only that the
P-first fibration is the one directly measurable from M(P).

### 3.3 Against the generic case

| | base | fiber | total |
|---|---|---|---|
| sedenion (S) | 11 | 3 | 14 |
| generic bilinear | 14 | 0 | 14 |

S's zero-divisor locus is **codimension 4** in P, not codimension 1, and its
annihilator is **4-dimensional**, not 1-dimensional. Same total, opposite
shape.

---

## 4. Proposed Stage 0 statistic — annihilator dimension

**Measure:** numerical rank deficiency of M(P) at variety points, S and X.

- **S: exactly 4.** Not a prediction. Triple-sourced (§3.2) before measurement.
- **X: 1** if the shuffle behaves generically. Unknown — this is the test.

Against the §7.2 power-check requirement:

- Binary and structural, in the style of the bilaterality discriminator that
  worked in §2.6 — not a contestable continuous magnitude.
- **Power demonstrable before freezing**, because the S value comes from
  theorems rather than from a pilot. This is the first candidate criterion
  that meets §7.2 without spending compute to establish separation.
- Zero compute. Should reuse the code that produced the dimension-14 figure.

**Bearing on HANDOFF §3.1.** If X's fiber is 1-dimensional where S's is 3, the
target in Q-space is codimension-3 thinner for X, giving "X can't reach low r²"
a mechanism with no dependence on conditioning. That would leave §3.2's
conditioning account and this one as two live alternatives, each needing its
own control. If X also measures 4, the alternative dies for one rank
computation. **Either way this runs before the §3.1 global restart search**,
since it changes what that search is testing.

---

## 5. §8.4.7 — the real gap is unread citations, not missing ones

Recommend §8.4.7 gain an item, worded against the demonstrated failure rather
than the one rev. A invented:

> **4. Abstract-level entries are being carried as if read.** Koebisu
> (arXiv:2512.13002) was cited in §3 from 2026-07-20 and in §8.3 from
> 2026-07-23, in both cases at abstract level. Its Corollary 3.8 was
> independently rediscovered numerically and written into `PHASE5_PLAN.md`
> §3.4 as a novel lead (corrected 2026-07-27). Every §3 supporting-literature
> entry should be marked **read** or **abstract-only**, and no entry may be
> relied on for a novelty claim while marked abstract-only.

This is a different failure mode from §8.4.7 items 1–3, which are all
*coverage* gaps closable by more search. This one is not: the search already
succeeded. More keyword queries will not fix it.

---

## 6. Minor

Koebisu §5: on a 3-dimensional purely imaginary cyclic slice,
D₂(X,Y,Z) = 4(XY + YZ + ZX)², so the slice of the zero-divisor set is the cone
XY + YZ + ZX = 0 — an A₁ singularity, quadratic form of signature (1,2),
origin the unique singular point on that slice. Not load-bearing for ZDA; one
line in the expanded §3 entry.

Koebisu bibliographic detail for §3: v1 15 Dec 2025, v2 26 Mar 2026; v1 was
titled *Singular Structures and Geometric Holonomy in the Zero–Divisor Set of
the Sedenions*. Single author, no institutional affiliation.

---

## 7. Verification list for Claude Code

1. Confirm the repo's dimension-14 figure is the **normalized**-pair
   dimension. §3.2 depends on it. Review §3 says normalized; confirm the repo
   measured the same object.
2. Rank-deficiency measurement (§4). Expect exactly 4 for S; report X.
3. Confirm line numbers 188, 190–191, 475 against the current
   `PRIOR_ART_REVIEW_zda.md` before quoting them in any commit message.
4. §3.2's V₂(ℝ⁷) ≅ G₂/SU(2) step is chat-side reasoning, not quoted from
   either paper. Re-derive independently before it is relied on.
