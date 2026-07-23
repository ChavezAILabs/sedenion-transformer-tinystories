# Phase 4 "Open Yard" — kernel scoping memo (2026-07-19)

Status: pre-spec mathematical scoping. Everything labeled VERIFIED below was
checked numerically against an independent NumPy implementation of the
Cayley–Dickson construction under the Baez convention
((a,b)(c,d) = (ac − d̄b, da + bc̄); conjugate (a,b)̄ = (ā, −b)), built from
scratch for this memo — i.e., it cross-checks `sedenion_kernel.py` rather
than importing it. The independent build reproduces the repo's ground
truth: Pattern 2 (P = e3+e12, Q = e5+e10) is bilateral (PQ = QP = 0
exactly) and the collapse identity (aP+bQ)(bP+cQ) = −2b(a+c)e₀ holds to
1e-10 over 200 random (a,b,c). VERIFIED.

Motivating diagnosis from Phase 3 (RESULTS_phase3.md §3, §5): the opt-in
gate was sampled by SGD and priced at perturbative magnitude. Phase 4 asks
whether zero-divisor structure matters when it is load-bearing in the
similarity computation — no parameter setting recovers standard attention.
This memo determines which kernels actually have that property. Two naive
designs die on the page; one survives with an unexpected bonus.

## 1. Four kernel candidates

Per head, queries and keys are mapped by learned projections either to
plane coordinates (a_i, c_j scalars on the fixed (P,Q) plane, with b a
learned or fixed plane constant) or to full 16-component sedenion
coordinates q_i, k_j. ⊗ denotes sedenion multiplication; ⟨·⟩₀ the e₀
(real) component.

### K1 — linear collapse kernel: s_ij = ⟨(a_iP+bQ) ⊗ (bP+c_jQ)⟩₀ = −2b(a_i+c_j)

**Verdict: degenerate. Dead on arrival.** The score is additively
separable in query and key. Softmax over keys eliminates the query term:
softmax_j(−2b·a_i − 2b·c_j) = softmax_j(−2b·c_j), identical for every
query. VERIFIED (attention rows numerically identical across distinct
queries). The direct "consolidate the collapse identity into the logit"
design produces query-independent attention — a static key-salience model,
not an attention mechanism. Any multi-plane linear combination
Σ_k β_k(a_i^k + c_j^k) is a sum of separable terms and is equally dead.
This is the single most important negative result in the memo: the obvious
Phase 4 design does not work, for a reason visible only once you push the
algebra through the softmax.

### K2 — modulus kernel: s_ij = ±|a_i + c_j| (per plane, multi-plane additive)

**Verdict: viable but weak; nonsmooth.** The nonlinearity breaks
separability (VERIFIED: attention rows differ across queries). With the
negative sign, the score is maximal exactly on the annihilation manifold
a_i + c_j = 0: the model attends *to* annihilation-aligned pairs; positive
sign attends away. This is Phase 3's gate feature promoted to the sole
score — genuinely mandatory. Two costs: (i) per plane, query–key
interaction passes through a single scalar, so expressivity is that of a
sum of 1-D matching kernels; (ii) the score has a gradient kink on the
manifold — exactly where the interesting behavior lives. Trainability risk
where it matters most.

### K3 — norm-ratio kernel: s_ij = g(r_ij²), r_ij = |q_i ⊗ k_j| / (|q_i||k_j|)

**Verdict: the viable mandatory kernel — and the central result of this
memo.** Full 16-component projections; score any monotone g of the
normalized product norm (e.g. s = −γ·r² to attend toward the manifold, or
s = −γ·log(r²+ε) variants).

Three verified properties:

1. **Genuine query–key interaction in sedenions.** log r_ij is not
   additively separable (double-centered residual ≈ 0.14 on random pairs;
   separable kernels give exactly 0). VERIFIED. Random-pair r ranges
   ~0.46–1.23 — note r > 1 occurs: sedenion norm can *inflate* under
   multiplication, not just deflate. The manifold-facing tail and the
   inflation tail are both live features.
2. **Exact structural nulls on the ZD manifold.** For q on the plane and
   k at annihilating coordinates, |q ⊗ k| = 0 exactly. VERIFIED. No
   parameter setting removes these nulls — they are properties of the
   multiplication, not of the weights. This is "open yard" in the strict
   sense.
3. **Smooth at the manifold.** r² scales as ε² for coordinates ε away
   from annihilation (VERIFIED: 1.96e-2 / 1.96e-4 / 1.96e-6 at
   ε = 1e-1/1e-2/1e-3), so the r² score is differentiable with vanishing
   gradient at the null — no K2-style kink. Trainability-friendly where
   K2 is not.

### K4 — structured bilinear readout: s_ij = w · (q_i ⊗ k_j), learned w ∈ R¹⁶

**Verdict: soft-optional — Phase 3's trap in disguise; keep only as a
diagnostic instrument.** Since the product is bilinear, s_ij = q_iᵀM(w)k_j
with M(w) = Σ_m w_m T[:,:,m]: a 16-parameter family spanning a 16-dim
subspace of the 256-dim bilinear forms (rank VERIFIED = 16). But the
family contains w ∝ e₀, and M(e₀) = diag(1,−1,…,−1) (VERIFIED) — a fixed-
signature form absorbable by the projections, i.e., a dense-attention-
equivalent point *inside the family*. SGD can walk to it. This is not an
open yard; it is a yard with a gate at coordinate e₀. Its redemption is as
an instrument: initialize w away from e₀ and log the trajectory — a direct
measurement of whether SGD migrates to the dot-product point when
ZD-structured scores are equally available in-family. Secondary variant at
most, never the headline.

## 2. The central mathematical result: the octonion control is a theorem, not a run

In octonions (CD3, the last division algebra) the norm is exactly
multiplicative: |x ⊗ y| = |x||y| (VERIFIED to 1e-9 over 500 random
pairs). Therefore the K3 ratio r_ij ≡ 1 identically, log r is exactly
separable (VERIFIED: double-centered residual = 0 to machine precision),
and octonion K3 attention is provably query-independent — degenerate in
precisely the K1 sense.

Read that forward: **in the norm-ratio kernel, all query–key interaction
is the deviation from norm multiplicativity — which is exactly the
zero-divisor phenomenon.** The interaction doesn't merely *include* ZD
structure; it *is* ZD structure, in the precise sense that the same kernel
over the largest zero-divisor-free CD algebra has no interaction at all.

Consequences for the spec:

- The planned empirical H4b (sedenion vs octonion at matched width)
  dissolves: the octonion arm is mathematically degenerate, so running it
  tests nothing about learning. Its information content is captured by a
  **query-independent (static-salience) attention baseline** — same
  architecture, scores depend on keys only — which prices "no interaction"
  empirically and costs one variant, not a second algebra implementation.
- ZD-specificity within K3 must instead be tested V3-style: a **rotated /
  random-frame K3** (conjugate the multiplication by a random orthogonal
  map, or equivalently randomly rotate the projection targets) and a
  **shuffled-tensor K3** (V4-style; note a shuffled T has its own
  accidental null structure, which must be characterized before the run,
  not after). The shuffled-T subspace is genuinely different from the true
  one (max principal-angle cosine ≈ 0.44, VERIFIED on one draw; per-draw
  characterization needed).

## 3. A second closed door, recorded so nobody reopens it

"Full hypercomplex attention" with the conjugated real-part score
Re(q ⊗ k̄) is *exactly* Euclidean dot-product attention: Re(x ⊗ ȳ) = Σx_iy_i
in every CD algebra (VERIFIED to 1e-10). The unconjugated Re(q ⊗ k) is the
fixed-signature form diag(1,−1,…,−1) — absorbable by learned projections,
hence the same model class as dense attention. **Zero-divisor structure is
invisible to any real-part bilinear score.** Any future proposal of
"sedenion attention via real parts" should be answered with this paragraph.

## 4. Positional encoding inside the algebra

Left-multiplication by any basis element e_i is a signed permutation of
coefficients, hence orthogonal (VERIFIED for all 16). Composed
basis-plane rotations therefore give an isometric, RoPE-analogous
positional action *within* the algebra — position can enter without a
parallel non-algebraic channel, preserving the mandatory property. Details
(which planes commute with the (P,Q) structure, i.e., which rotations
preserve vs. traverse the manifold) are the main open derivation for the
spec; a rotation that moves pairs across the annihilation manifold as
relative position changes would make "structural ignoring" position-
dependent, which is either a feature (learned locality) or a confound, and
must be settled on paper first.

> **Settled 2026-07-19 (Claude Code, `phase4_positional.py`):** all 15
> candidate generators satisfy L_i² = −I (valid rotation groups), but
> **e8 — the CD doubling generator — is the unique basis generator whose
> rotation R_8(θ) = cos θ·I + sin θ·L_8 gives the relative-position
> property for the K3 kernel** (|R_8(θ)q ⊗ R_8(φ)k| depends only on φ−θ,
> verified on random pairs, tol 1e-9). Under R_8, annihilating Pattern-2
> pairs remain exactly null at *every* relative offset: e8-position never
> traverses the manifold, so "structural ignoring" is position-independent
> by construction — neither feature nor confound; position and ZD
> structure decouple exactly. All other generators fail the relative
> property and (except 6, 8, 14 at shared phase) break nulls; all except
> e8 oscillate across the manifold under relative phase.
>
> **Frequency structure (settled same day):** per-plane multi-frequency
> R_8 (RoPE-style blocks in L_8's 8 invariant planes) **fails** the
> relative property badly (deviation ~3.4 — the block rotation is no
> longer product-compatible), while a single frequency per head,
> R_8(ω_h·pos), holds it **exactly** (ω_h·pos is a reparameterized R_8
> angle). The positional design is therefore forced: one frequency per
> head, RoPE's frequency ladder distributed across heads rather than
> within a head. Consequence for the spec: per-head positional resolution
> is single-scale; head count bounds the frequency spectrum.

## 5. What this does to the Phase 4 spec skeleton

Primary variant: **K3-sedenion** (norm-ratio score, sign/shape of g fixed
in the spec after the positional derivation). Hypotheses to pre-register,
decision rules fixed before any run:

- **H4a:** K3-sedenion vs param- and FLOP-matched dense baseline on val
  loss, and vs the length-generalization record of the PHM family
  (ppl@1024 ≈ 8.2–8.8 from Phase 3 — the bar an "ignoring geometry" story
  must at least approach to be interesting).
- **H4b′ (specificity, restructured):** K3-sedenion vs random-frame K3 and
  vs shuffled-tensor K3 (with pre-run null-structure characterization of
  the shuffled tensor); the query-independent baseline prices the
  no-interaction floor. The octonion arm is replaced by the §2 theorem.
- **H4c (mechanism, falsifiable pre-loss):** heads' learned coordinates
  migrate toward the annihilation manifold during training. Mandatory
  logged diagnostic: per-head distance-to-manifold traces (the Phase 4
  analogue of β traces), plus attention-entropy per head. If coordinates
  never approach the manifold, the mechanism story dies regardless of
  loss numbers — same epistemic role the β traces played in Phase 3.

Validation-chain deltas to relay to Claude Code (suggestions, to be
verified against the chain, not applied directly):

1. T4 (β=0 ≡ baseline) cannot survive by construction — mandatory means no
   dense-equivalent parameter setting. Replacement anchors: (i) structural-
   null test — kernel evaluates to 0 (fp tolerance) on annihilating pairs
   drawn from the 336-pair table in `sedenion_kernel.py`; (ii) K1
   degeneracy test — the linear readout of the same pipeline must produce
   query-independent rows (guards against silently reintroducing a
   separable score); (iii) octonion-degeneracy test — swapping T16 → T8
   must produce query-independent attention (a strong end-to-end check
   that the implementation's interaction really comes from the algebra);
   (iv) causality (T6 analogue) and NumPy-mirror-first, unchanged.
2. Numerical hygiene for r near 0: score gradients vanish smoothly, but
   log-form scores need the ε; fp16/TF32 behavior near the manifold needs
   a targeted test before any GPU run.
3. Trainability smoke (Shakespeare-scale) before any grid, watching H4c's
   distance-to-manifold traces from step 0.

## 6. Honest prior, restated

Phase 3's precedent stands: offered the ZD direction for free, SGD priced
it near zero. K3 removes the pricing mechanism — the model cannot decline
the geometry, only choose where in it to place tokens. The most likely
outcome remains a null (coordinates drift nowhere near the manifold;
K3-sedenion ≈ random-frame K3 ≈ dense). What makes the null worth buying
is §2: for this kernel family, "interaction from zero divisors" is not a
hypothesis about the result but a theorem about the design, so even a
clean negative becomes a statement that *the one kernel whose interaction
is purely ZD-generated is not preferred by SGD at this scale* — a sharper
sentence than Phase 3 could earn. IGP24 provenance for the consolidation
logic (lowest-discriminant priority → fewest-DOF integration) should be
KSJ-captured with this memo's date.

## 7. Numerical verification log

All checks in this memo were run against the independent T16/T8 structure
tensors (Baez, built by 4- resp. 3-level CD doubling from R): Pattern 2
bilaterality; collapse identity (200 random triples, 1e-10); K1 row-
identity; K2 row-distinctness; K3 sedenion double-centered log-ratio
≈ 0.139 vs octonion 0.0; structural null |q ⊗ k_null| = 0.0 with r² ~ ε²
approach; Re(x ⊗ ȳ) = ⟨x,y⟩ (200 pairs, 1e-10); M(e₀) = diag(1,−1,…,−1);
rank(span{T[:,:,m]}) = 16, shuffled-T principal-angle overlap ≈ 0.44 (one
draw); all 16 basis left-multiplications orthogonal. Independent
implementation ≠ repo-verified: Claude Code should reproduce each check
against `sedenion_kernel.py`'s tensor before any of this enters the spec.

## 8. Repo verification addendum (Claude Code, 2026-07-19)

**Gate passed.** Every §7 claim reproduced against `sedenion_kernel.py`'s
tensor by `verify_phase4_memo.py` (repo root, rerunnable) — all checks OK.
Notes from the reproduction:

- K3 double-centered log-ratio residual came out 0.095 on this RNG vs the
  memo's 0.139 — it is a draw-dependent statistic; both are ~15 orders
  above the separable floor (1e-16). r observed in [0.56, 1.31], inflation
  tail confirmed.
- ε-scaling of r² at the manifold confirmed quadratic asymptotically
  (decade ratios 101.4, 100.1 at ε = 1e-2/1e-3/1e-4; at ε = 1e-1 the key
  norm in the denominator is still moving, so the first decade reads ~115
  — not a deviation, just pre-asymptotic).
- Shuffled-T overlap measured with the repo's **actual V4 recipe**
  (`shuffled_left_mult_matrices`: L_0 = I, random signed permutations):
  max principal cosine 0.40–0.47 over 5 draws — the memo's single-draw
  0.44 is typical and the statistic is stable, though per-draw null-
  structure characterization for the chosen run seed(s) remains required
  (§2).
- §4's open derivation is settled: see the boxed note in §4
  (`phase4_positional.py`) — positional action is R_8, uniquely; no
  manifold traversal; within-head multi-frequency fails, per-head single
  frequency exact (frequency ladder goes across heads).
- **A third closed door, found during harness build (2026-07-19): the
  random-frame K3 control (§2's "rotated / random-frame K3") is vacuous.**
  Conjugating the multiplication by orthogonal O gives
  |O^T·mult(Oq,Ok)| = |mult(Oq,Ok)| — exactly the S-family score after
  W → OW, which learned full-rank projections absorb (verified to
  1e-10; the conjugated positional generator transports likewise,
  1.8e-14). The only non-vacuous hybrid — conjugated tensor with the
  *true* R_8 rotations — breaks the relative-position property (dev
  ~0.9): confounded, not a control. Phase 3's V3 worked only because its
  frame vectors were fixed constants. In a mandatory kernel with learned
  projections there is no V3 analogue; ZD-specificity is carried by the
  shuffled-tensor control X (not conjugation-equivalent: principal
  cosine ≈ 0.45), the Q0 no-interaction floor, and the §2 octonion
  theorem. Spec v0.3 drops variant R accordingly.
