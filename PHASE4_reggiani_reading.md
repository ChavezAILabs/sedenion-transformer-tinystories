# Reggiani reading — findings and a proposed X-inequivalence certificate

**Date:** 2026-07-21. **Author:** Claude Code, at the owner's explicit
direction this session (this reading was flagged in `PHASE4_gate5_outcome.md`
§4 item 2 as *not* to start autonomously, since it feeds a pre-registration
decision — the owner has now asked for it directly).

**Source:** Reggiani, S., "The geometry of sedenion zero divisors,"
arXiv:2411.18881 (v5, 2024-11-26). Read from the paper's own LaTeX source
(`arxiv.org/e-print/2411.18881`), not a rendered PDF or an AI-summarized
pass of one — a first WebFetch pass on the abstract page misreported
dim ZD(𝕊) as 12 (it is 11: dim G₂ − dim SU(2) = 14 − 3); the raw source
was pulled specifically to avoid propagating that kind of error into a
pre-registration decision.

## 1. What the paper proves (verified quotes, §2.2–§3 of the source)

- $\mathcal Z(\mathbb S) = \{(u,v)\in\mathbb S\times\mathbb S : \|u\|=\|v\|=\sqrt2,\ uv=0\}$
  is **diffeomorphic to $G_2$** (Moreno 1998, cited), and Reggiani's new
  result: **isometric to $G_2$ with a naturally reductive left-invariant
  metric**, total space of a Riemannian submersion over $G_2/\mathrm{SO}(4)$
  (the symmetric space of quaternion subalgebras of $\mathbb O$), fibers
  locally isometric to a product of two round 3-spheres of different radii.
  $\dim \mathcal Z(\mathbb S) = 14$.
- $\ZD(\mathbb S) = \{u : (u,v)\in\mathcal Z(\mathbb S) \text{ for some } v\}$
  is isometric to the Stiefel manifold $V_2(\mathbb R^7) = G_2/\mathrm{SU}(2)$,
  $\dim = 11$.
- **Proposition 2.1** (the operative characterization): $(a,b)\in\mathbb S$
  is a zero divisor iff $a,b$ are imaginary octonions with $\|a\|=\|b\|\neq0$
  and $a\perp b$.
- $\dim(\ann u)$: Moreno proves $\equiv 0 \bmod 4$; Biss–Dugger–Isaksen
  prove $\le 2^n-4n+4$, which at $n=4$ (sedenions) is exactly 4 — so for
  every sedenion zero divisor, $\dim(\ann u) = 4$, pinned exactly, not
  just bounded.
- Appendix Table 1 lists **84 explicit "standard zero divisor" pairs**
  $(e_i+e_j,\ e_k\pm e_l)$, chosen as convenient origins for the metric
  computations (any zero-divisor pair works, by $G_2$-homogeneity — these
  are just the ones that make the linear algebra easiest).

## 2. Repo cross-checks (`verify_phase4_reggiani.py`, all green)

Transcribed Table 1 verbatim from the LaTeX source (not the PDF) and
checked it against `sedenion_kernel.cd_mult`:

1. **All 84 published pairs are exact zero divisors under this repo's
   kernel** (`|uv| < 1e-12` for every row) — the repo's Baez-convention
   `cd_mult` and Reggiani's Cayley-Dickson formula agree completely; no
   convention mismatch.
2. **The project's `ZD_PAIR` — `(e3+e12, e5+e10)`, `zda_layers.py:105` /
   `zda_grid.py:137` — is literally row 52 of Reggiani's own table.** This
   was chosen pre-Phase-3 as KSJ "Pattern 2" from the repo's own exhaustive
   search, independent of this paper; finding it verbatim in a published
   classification is a strong, previously-unknown corroboration that the
   project's canonical pair isn't an edge case — it's one of the 84
   the literature itself treats as canonical.
3. **The repo's "336 pairs / 84 left-null two-blades" numbers
   (`sedenion_kernel.py` check 4, `phase4_shuffledT_nulls.py`) are now
   *derived*, not just observed:** Prop 2.1 gives $84 = 42\ (i,j)$
   combinations $\times\ 2$ signs directly (i ranges the first CD-half
   1–7, j the second CD-half 9–15, excluding the 7 diagonal cases
   $i=j-8$ where $a=b$ isn't orthogonal to itself). And $336 = 84\times4$
   is exactly $\dim(\ann u)=4$: every one of the 84 canonical two-blades
   has precisely 4 signed two-blade partners among the 210 candidates.
   Confirmed by direct computation, not just formula-matching.

## 3. A new X-inequivalence certificate (`phase4_reggiani_certificate.py`)

`phase4_shuffledT_nulls.py` (SS9 item 1, marked DONE 2026-07-19) already
established that shuffled tensor X has fewer exact two-blade null pairs
than the true tensor (66–96 vs. 336) and is diffusely near-singular
(σ_min(A_x) over random directions ~12× smaller). Both are *statistical*
distinctions — they compare counts and distributions, not the geometric
character of the null set itself.

Reggiani's theorem licenses a sharper, non-statistical test: **because
$\mathcal Z(\mathbb S)$ is diffeomorphic to a smooth compact manifold
($G_2$), it has no singular points, anywhere.** At any zero-divisor pair
$(u,v)$, the differential of $F(u,v)=uv$, restricted to the tangent space
of $S^{15}\times S^{15}$ at $(u,v)$ (30 real dimensions: $u^\perp\oplus
v^\perp$), is a $16\times30$ matrix $J = [R_v|_{u^\perp},\ L_u|_{v^\perp}]$.
By the implicit function theorem, $J$ full rank (16) $\Rightarrow$ the
null set is *locally a smooth 14-manifold* there; rank-deficient $J$
$\Rightarrow$ a genuine singularity (locally higher-dimensional or
non-manifold). Reggiani's theorem forces $J$ full-rank at **every** point
of $\mathcal Z(\mathbb S)$ — no exceptions are possible for the true
tensor. No comparable theorem exists for an arbitrary shuffled tensor.

**Result, run on the pinned seeds (1337/1338/1339, same draws as
`phase4_shuffledT_nulls.py`):**

| tensor | n null pairs | σ_min(J) min | median | max | singular points (σ_min<1e-6) |
|---|---|---|---|---|---|
| TRUE T16 | 336 | **1.41421 (=√2, exactly, all 336)** | 1.41421 | 1.41421 | **0/336** |
| X seed 1337 | 96 | 7.2e-18 | 0.753 | 1.028 | 3/96 (3.1%) |
| X seed 1338 | 92 | 0.0 | 0.746 | 1.060 | 7/92 (7.6%) |
| X seed 1339 | 66 | 4.2e-18 | 0.730 | 1.078 | 6/66 (9.1%) |

Two findings, both sharper than the existing characterization:

1. **Homogeneity fingerprint:** the true tensor's σ_min(J) is not just
   nonzero — it is *exactly* constant (√2) across all 336 known null
   pairs, to floating-point precision. This is the numerical signature of
   $G_2$ acting transitively and isometrically on $\mathcal Z(\mathbb S)$
   — every point looks geometrically identical, as a homogeneous space
   requires. X's σ_min values are scattered (median ~0.73–0.75, wide
   range) — its null set has no such symmetry, consistent with it being
   an accidental, non-homogeneous set rather than a group orbit.
2. **Genuine singularities:** 3–9% of X's found null pairs are exact
   singular points of the Jacobian (σ_min ~1e-17, i.e. machine zero) —
   points where the null set is *not* locally a manifold. The true tensor
   has zero such points, provably (the theorem forbids them). This is a
   geometric inequivalence, not a statistical one: X's null structure
   isn't just sparser or differently distributed, part of it is a
   different *kind* of object (singular) than anything on $\mathcal
   Z(\mathbb S)$ could ever be.

## 4. Disposition — not yet folded into the spec

This is additive to, not a replacement for, `phase4_shuffledT_nulls.py`
(that item stays DONE as-is; its numbers are unchanged and still cited).
`phase4_reggiani_certificate.py` is a new, standalone script — nothing
existing was edited.

**Proposed but not yet applied:** add a sentence to
`ZDA_phase4_spec.md` §9 item 1's "Interpretation constraint" — currently
"X is not null-matched... geometry difference, in either direction" —
noting that X additionally contains genuine geometric singularities that
the true tensor's null set providably cannot have, strengthening (not
changing the direction of) the existing caveat. This is a spec edit and
therefore held for the owner's go-ahead rather than applied here,
consistent with why this reading itself was deferred.

**Still open** (from `PRIOR_ART_REVIEW_zda.md` §7, unchanged by this
reading): the two remaining targeted literature searches (non-associative-
algebra attention / degenerate Clifford metrics; PHM-vs-length-
generalization) and the forward-citation traversal from Zoology and
Reggiani.
