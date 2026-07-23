# §8.4 — Hypercomplex-ML branch, and the score-vs-projection boundary (2026-07-23)

**Status: MERGED into `PRIOR_ART_REVIEW_zda.md` §8.4 on 2026-07-23 by Claude
Code**, after independently verifying every load-bearing claim against the
source paper (arXiv:2405.07024) directly — Table 1's missing sedenion row,
the §2.1 zero-divisor quote, the §4.1 PHAtt equation, and the §4.2
equivariance-loss quote all confirmed verbatim; the two Tier-2/3 adjacencies
(arXiv:2606.20547, arXiv:2602.10195) confirmed real with matching core
claims. Verification note lives in the merged section, not here. This file
is kept as the original delivered draft for the record; the canonical copy
is now in `PRIOR_ART_REVIEW_zda.md`.

**Status (original, historical):** draft for Claude Code to verify and merge into `PRIOR_ART_REVIEW_zda.md`.
Additive to §8.3; does not modify it. Cross-references flagged inline.
**Author:** Claude (claude.ai chat), parallel assignment while Claude Code
ran the GATr/Clifford re-root.

**Method and its limits (state this before anything else).** Web search
over titles, abstracts, and — where retrievable — full texts. This is
*weaker* than §8.3's Semantic Scholar traversal and weaker than a
structured citation query. It is topic-directed reading with one
full-text confirmation, not exhaustive coverage. Every negative below
is absence of evidence. The one positive finding (§8.4.3) is
full-text-verified and is the only claim here that does not depend on
search coverage.

---

## 8.4.1 Why this branch was invisible to §8.3

§8.3 traversed forward from Zoology and Reggiani. No paper in the branch
below cites Zoology — different community, different vocabulary, no
shared task family — so forward traversal from that root could not
reach it by construction. This is the concrete instance of §8.3's own
power caveat, and it should be cited there as such.

The entry point was Reggiani (arXiv:2411.18881) itself, which cites
`[SA20]` as a machine-learning application of sedenions. Resolving that
reference opens the branch. §8.3's forward pass from Reggiani found one
citing paper; the *backward* direction found the live literature. Noted
as a methodology lesson: for a sparse node, backward citation and
author-level search have power where forward traversal has none.

## 8.4.2 The branch

Three clusters, one research group lineage (Khalifa University;
Sapienza/Imperial for the framework paper).

1. **Saoud & Al-Marzouqi**, *Metacognitive Sedenion-Valued Neural
   Network and Its Learning Algorithm*, IEEE Access 8:144823–144838,
   2020. Sedenion-valued network for time-series forecasting; the
   metacognitive controller keys on sedenion magnitude and the 15
   phases. Extends the same authors' prior quaternion and octonion
   work. Sedenions as a phase-decomposition substrate.

2. **Bojesomo, Liatsis & Al-Marzouqi**, *Traffic flow prediction using
   Deep Sedenion Networks*, arXiv:2012.03874 (Traffic4cast 2020), and
   *Deep Hypercomplex Networks for Spatiotemporal Data Processing*,
   IEEE Signal Processing Magazine 41(3):101–112, 2024. Sedenion U-Net
   and a general hypercomplex spatiotemporal program. Component
   allocation only: sixteen time stamps mapped one-per-sedenion-
   component, with octonion and quaternion runs as lower-order
   comparisons. Stated motivation throughout is parameter efficiency
   (a 16× reduction). A 2022 follow-on describes sedenion *token
   mixing* in a transformer; the algebra is in the mixing layer, not
   in any pairwise score.

3. **Comminiello, Grassucci, Mandic & Uncini**, *Demystifying the
   Hypercomplex: Inductive Biases in Hypercomplex Deep Learning*, IEEE
   Signal Processing Magazine, 2024 (arXiv:2405.07024). The field's own
   theoretical framework paper, by the PHM/PHNN lineage. Full text
   retrieved and read. This is the load-bearing document for §8.4.3–5.

## 8.4.3 The positive structural finding (full-text verified)

This subsection is the evidentiary basis for the framing recommendation
adopted in §8.3, and does not depend on search exhaustiveness.

Three observations from arXiv:2405.07024, in ascending order of weight:

**(a)** The paper's opening premise attributes hypercomplex deep
learning's advantages to the properties of *division algebras* over
real vector spaces. The field grounds its own success account in
precisely the property sedenions lose.

**(b)** §2.1 names the loss once and moves on: all Cayley-Dickson
algebras at n ≥ 16 have zero divisors, and each doubling forfeits an
algebraic symmetry — commutativity at the quaternions, associativity at
the octonions, norm-multiplicativity at the sedenions. Framed as a
ledger of what is given up. Zero divisors do not reappear in the paper.

**(c) The decisive one.** Table 1 ("Algebraic convolution and
properties for domains") enumerates real, complex, quaternion,
tessarine, dual quaternion, and **octonion** — and stops. There is no
sedenion row, in a paper that introduces sedenions two pages earlier.
Correspondingly, §3.3 ("Algebraic Bias") derives usable inductive
biases from quaternion non-commutativity, octonion non-associativity,
and Clifford anticommutation. Never from annihilation.

**Therefore:** "the Cayley-Dickson ML literature stops at the octonions
and treats the onset of zero divisors as the reason not to go further"
is not an inference from our reading. It is the field's own framework
paper, in a table, in 2024. Cite Table 1 specifically. This converts
§8.3's headline framing from an assertion into a checkable claim.

## 8.4.4 The score-vs-projection boundary — sharper novelty statement

§4.1 of arXiv:2405.07024 specifies **parameterized hypercomplex
self-attention (PHAtt)**: Q, K, V are produced by PHM layers, then the
attention score is the ordinary scaled dot product softmax(QKᵀ/√d_k).
The underlying PHM construction is Zhang et al., ICML 2021 — the basis
of our own V1.

The generalization this licenses:

> Across the hypercomplex-attention literature, the algebra lives in the
> **projections** and the **score** remains a Euclidean dot product.
> K3 moves the algebra into the score.

Recommended as the load-bearing sentence of the related-work section.
It is narrow, verifiable against a specific equation, and survives the
discovery of further hypercomplex transformers — any new one that puts
algebra in the projections falls on the far side of the same line.

Note this composes with the §8.3 GA finding rather than duplicating it:
the GA family scores with the grade-0 projection of the geometric
product (dense-equivalent per our own memo, hence Tier 3 *by proof*),
and the hypercomplex family scores with the plain dot product after
algebraic projection. Two different literatures, same boundary, and K3
is on the other side of it in both.

## 8.4.5 Independent support for the V1/V4 outcome (methodology note)

§4.2 of arXiv:2405.07024 reports that PHNNs shed domain-specific
algebraic properties while retaining the Hamilton-*like* weight-sharing
structure, and confirms they lose the rotation and translation
equivariance that genuine quaternion and dual-quaternion models
possess. The field's own diagnosis of PHM is that the benefit is weight
sharing, not algebra.

This is convergent, from an independent direction, with
`RESULTS_phase2_smoke.md` §6: V1's length-generalization win was
matched by V4 and attributed to weight sharing. We found it empirically
via a control; they report it analytically as a general property of the
construction.

**Suggested use:** cite in the Phase 2 write-up. It upgrades the V1/V4
result from an isolated negative to a replication of a documented
property of PHM — a stronger and more citable thing to report. It also
retroactively supports the decision to build V4 at all.

## 8.4.6 Adjacency found outside this branch — Tier 2

**Lie-Algebra Attention** (arXiv:2606.20547, June 2026) — the closest
formal match found in any pass to date, and the item most likely to be
raised in review. Tokens are bare matrix Lie group elements; the
attention score is the negative squared algebra norm of the relative
pose, s_ij = −‖log(g_i⁻¹g_j)‖²_λ/τ, closed-form rather than learned.
Same shape as K3's s = −γ·r², both on a relative quantity, both
closed-form. The paper also positions itself explicitly against RoPE
and LieRE, placing it in the relative-position neighborhood as well.

**Distinguishing paragraph (draft — needs a repo sanity check):** their
norm is taken on a *group*, so it vanishes only on the diagonal
g_i = g_j; the attend-here set is "same pose," and the score is a
proximity kernel. K3's r² is a *product* norm on an algebra possessing
zero divisors, so it vanishes on a non-trivial variety at non-equal,
non-zero arguments; the attend-here set is the ZD variety. Same
functional form, categorically different zero sets. This warrants a
paragraph in related work, not a bare citation.

**Versor** (Huy & Hirst, arXiv:2602.10195, 2026) — conformal GA
Cl(4,1), recurrent rotor sequence architecture, benchmarked on
character-level WikiText-103 against a GATr baseline. Adjacent on the
sequence-modelling axis. Rotors are invertible; no annihilation, no
pairwise algebraic score. Tier 3, but cite — it is the nearest GA work
to our task family.

## 8.4.7 Open conditions (do not mark §8 closed)

1. **PHM forward traversal.** Zhang et al. (ICML 2021) is a
   heavily-cited node and the most likely remaining home for an
   unnoticed hypercomplex-attention variant. Not attempted here; too
   large for web search. Needs the Semantic Scholar API when reachable.
   This is the single largest remaining gap on this branch.
2. **Patent literature — entirely unexamined.** USPTO 12524654,
   "Metacognitive sedenion-valued neural networks" (same group as
   §8.4.2 item 1). Publication novelty and freedom-to-operate are
   different questions with different bars. Flagged only; no legal
   assessment offered or implied. Relevant only if the work is ever
   intended to support more than a paper.
3. Carried from §8.3: STAResNet abstract-level only; exhaustive
   GATr/Clifford citation counts pending API access.

## 8.4.8 Convention flag

Sources in this sweep report sedenion zero-divisor counts and exemplar
pairs inconsistent with the repo: 84 rather than 336, and pairs such as
(e3+e10)(e6−e15) and (e5+e10)(e6+e9). Almost certainly a combination of
counting convention (ordered vs. unordered, normalized vs. not) and
multiplication-table convention. Per standing rule: flagged, not
reconciled, not averaged. **Any of these figures must be converted into
Baez and into the repo's counting before appearing beside a repo number
in any draft.**

---

## Net position after §8.3 + §8.4

Three neighborhoods, three methods, one consistent posture toward
annihilation:

- **Recall-task literature** (§8.3, forward traversal from Zoology):
  never raises the vocabulary.
- **Clifford / geometric-algebra attention** (§8.3, re-rooted): scores
  with the grade-0 projection — dense-equivalent by our own memo — and
  its own authors describe the degeneracy of the inner product as a
  limitation to be remedied by mapping into CGA.
- **Cayley-Dickson hypercomplex ML** (§8.4): tabulates algebraic
  properties only as far as the octonions, and names zero divisors once,
  as a loss.

**Claimable:** the mechanism was not found in three surveyed
neighborhoods, and the field's own framework paper documents the
boundary K3 inverts.

**Not claimable:** exhaustiveness. §8 remains open pending §8.4.7.
