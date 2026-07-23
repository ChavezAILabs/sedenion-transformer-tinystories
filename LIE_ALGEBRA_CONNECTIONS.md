# Lie algebra connections — G₂ and E₈, and the Canonical Six

**Date:** 2026-07-21. **Author:** Claude Code, at the owner's direction
("ensure that our own Lie algebra connections is not overlooked"),
following up on `e8_weyl_orbit_unification.lean` being placed in this
repo. This document exists so the connection isn't lost — it is not part
of the Phase 1–4 validation chain and touches nothing in it.

## 0. What this is and isn't

This project's `ZD_PAIR` (`e3+e12, e5+e10`, `zda_layers.py:105`) already
has a documented provenance: `RESULTS_phase2_smoke.md` §6 records it as
KSJ "Pattern 2" (AIEX-725), "the unique Canonical Six pair that is
bilateral in both the Cayley-Dickson and Clifford frameworks," and cites
`canonical_six_bilateral_zero_divisors_cd4_cd5_cd6.lean` (Lean 4,
`native_decide`, zero sorry stubs) as grounding.

**"The Canonical Six" is the owner's own separate, published work:**
Chavez, P., *"Framework-Independent Zero Divisor Patterns in
Higher-Dimensional Cayley-Dickson Algebras: Discovery and Verification of
The Canonical Six,"* Zenodo, concept DOI
[10.5281/zenodo.17402495](https://doi.org/10.5281/zenodo.17402495),
latest v1.3 (Feb 2026) DOI
[10.5281/zenodo.18793480](https://doi.org/10.5281/zenodo.18793480),
formal verification co-authored by Aristotle (Harmonic Math). Project
directory: `C:\dev\projects\canonical_six_publication\`. This document
does not edit anything in that repo — it records, on the ZDA side, what
that work implies for `ZD_PAIR`, and flags one thing (§4) that may be
useful to carry back the other direction.

## 1. What's newly verified here: the E₈ Weyl-orbit addendum (v1.3)

`e8_weyl_orbit_unification.lean` is v1.3's Addendum B (`canonical_six_
publication/repo/lean-formalization/`, zero sorry stubs, `native_decide`/
`decide +kernel`). It is **new relative to what `RESULTS_phase2_smoke.md`
§6 already cites** — that citation is to the bilateral-zero-divisor proof
(Addendum A), not this one.

**What the Lean file proves**, independently reproduced here in exact
rational arithmetic (`verify_e8_weyl_orbit.py`, `fractions.Fraction`
throughout — no floating point, mirrors the Lean `ℚ` type exactly):

- **Theorem 1a:** five specific 8-dimensional vectors $v_1,\dots,v_5$
  (given as explicit rational coordinates) all have squared norm 2 —
  i.e. they are roots of $E_8$ (the "first shell").
- **Theorem 1b:** $v_2 + v_3 = 0$ (an antipodal pair), connected by a
  single simple reflection $s_{\alpha_4}$.
- **Theorem 1c:** all five reduce to the same dominant weight
  $\lambda=(1,0,0,0,0,0,0,-1)$ under explicit Weyl-reflection sequences
  — i.e. **all five lie in a single $E_8$ Weyl orbit.**

`verify_e8_weyl_orbit.py` reproduces all three theorems exactly (every
assertion passes, no tolerance needed — rational arithmetic is exact) and
separately confirms, against **this repo's own `sedenion_kernel.cd_mult`**
(Baez convention), that all six $(P_i,Q_i)$ pairs the Lean file defines
are genuine bilateral zero divisors, and that **$P_2=e_3+e_{12}$,
$Q_2=e_5+e_{10}$ is exactly this project's `ZD_PAIR`.**

**Gap, stated plainly (checked, not assumed):** the Lean file defines
$v_1,\dots,v_5$ by their raw 8-dimensional coordinates directly — it does
not contain a formula deriving them *from* the sedenion pairs
$P_1,\dots,P_6$. So this file, taken alone, proves a real and
self-contained fact about five specific $E_8$ roots; it does not, within
itself, prove which sedenion structure they are "images of" or by what
map. (The paper text, not fetched here, may state this derivation — not
checked as part of this pass.) Read the "images of the P-vectors" framing
as the source repo's characterization of the result, verified as far as
this file goes, not independently re-derived end-to-end.

## 2. Triple convergence on the same pair

Three independent lines, none aware of the other two when each was
established, land on the identical pair $(e_3+e_{12},\ e_5+e_{10})$:

1. **This project**, pre-Phase-3 (2026-07-17): adopted as `ZD_PAIR` from
   KSJ "Pattern 2," on the strength of being the unique Canonical Six
   pair bilateral in both frameworks (`RESULTS_phase2_smoke.md` §6).
2. **The Canonical Six paper** (Chavez, Zenodo, independently of this
   project): pattern **59** / "S2," the same pair, established as one of
   six framework-independent patterns out of 168 sedenion zero divisors
   (3.6%) — see `canonical_six_publication/repo/README.md`.
3. **Reggiani** (arXiv:2411.18881, read this session — see
   `PHASE4_reggiani_reading.md`): the pair is literally row 52 of the
   paper's own Appendix Table 1 (the "84 standard zero divisors"),
   verified against this repo's kernel in `verify_phase4_reggiani.py`.

None of these three needed the other two to pick this pair; all three
picked it anyway. That's a reasonable, if informal, sanity signal that
$(e_3+e_{12}, e_5+e_{10})$ sits at a mathematically distinguished point
of the sedenion zero-divisor structure, not an arbitrary convenient
choice — worth a line in any future ZDA write-up that discusses why this
particular pair, beyond "it's a verified annihilating pair" (the spec's
stated minimum bar, `ZDA_experiment_spec.md` §2.3).

## 3. Two different exceptional groups, two different objects — don't conflate

Both $G_2$ and $E_8$ show up in this project's orbit now, for genuinely
different reasons, and the naming is a minefield for accidental
conflation:

- **Reggiani's $G_2$** is about the *continuous* zero-divisor variety
  $\mathcal Z(\mathbb S) = \{(u,v): \|u\|=\|v\|=\sqrt2, uv=0\}$ — proven
  diffeomorphic (isometric, in fact) to the 14-dimensional Lie group
  $G_2 = \mathrm{Aut}(\mathbb O)$, acting *transitively*. Any single
  zero-divisor pair, including `ZD_PAIR`, is one point on this
  homogeneous space; $G_2$ acts on the whole continuum, not on a finite
  subset.
- **This Lean file's $E_8$** is about a *finite, discrete* structure: five
  specific rational vectors, images (by an unstated-in-this-file map) of
  six specific two-blade sedenion pairs, shown to be $E_8$ roots forming
  one Weyl orbit. $E_8$'s Weyl group is finite (order ~696 billion) and
  acts on the *root lattice*, not on the continuous $\mathcal Z(\mathbb
  S)$ directly.
- **Unrelated to both:** `phase4_positional.py`'s $R_8(\theta)$ — the
  positional rotation generator used by K3 — is called "$e_8$" only
  because it's the 8th Cayley-Dickson basis vector (the doubling
  generator between octonions and sedenions, 0-indexed). It has nothing
  to do with the $E_8$ Lie group. Given this document exists specifically
  because "$E_8$" now appears in two unrelated senses in project-adjacent
  material, this is flagged explicitly so a future reader (or session)
  doesn't merge them.

## 4. One thing worth carrying back the other direction (not acted on here)

`canonical_six_publication/repo/lean-formalization/LEAN_README.md` lists
an open stub in `g2_family_24_investigation.lean` /
`master_theorem_scaffold_phase5.lean`: **"G₂ Lie-theoretic invariance —
requires G₂ representation theory in Mathlib."** Reggiani's paper (read
this session for unrelated reasons — the ZDA X-inequivalence certificate,
`PHASE4_reggiani_reading.md`) is exactly the missing piece of
literature: it gives $G_2$'s explicit action on $\mathcal Z(\mathbb S)$
and $ZD(\mathbb S) \cong V_2(\mathbb R^7) = G_2/\mathrm{SU}(2)$ in closed,
citable form. Whether it's enough to actually close that Mathlib-blocked
stub is a question for that project, not this one — noted here only
because it was found here, per this project's own discipline of writing
down connections when found rather than losing them. Not acted on in
`canonical_six_publication/` — that repo has its own publication
discipline (DOI'd, versioned) and edits there are out of scope for this
session unless asked.

## 5. Artifacts

- `verify_e8_weyl_orbit.py` (this repo) — independent exact-rational
  reproduction of the three Lean theorems, plus the bilateral-ZD /
  `ZD_PAIR` cross-check. Rerunnable, no Lean toolchain required.
- `e8_weyl_orbit_unification.lean` (this repo, copied in by the owner
  from `canonical_six_publication/repo/lean-formalization/` for this
  check) — the original Lean 4 source, Aristotle/Harmonic Math,
  `native_decide`/`decide +kernel`, zero sorry stubs.
