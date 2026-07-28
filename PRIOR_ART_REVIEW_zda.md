# Prior Art Review — ZDA / Phase 4

**Date:** 2026-07-20
**Author:** Claude Desktop (analysis lane)
**Scope:** literature bearing on (a) the dial task and gate-5 failure, (b) the
K3 kernel and hypercomplex attention, (c) sedenion zero-divisor structure,
(d) attention dials, (e) the Phase 3 length-generalization finding.
**Method:** web search, 2026-07-20. Coverage is good but not exhaustive;
treat absence of a result as "not found," not "does not exist."

---

## 0. Executive summary

Five findings, in descending order of actionability.

1. **The gate-5 failure is very likely a protocol gap, not capacity.** In the
   established benchmark for this exact task family (MQAR), *softmax attention
   solves it perfectly at model dimension 64 with two layers*, and is routinely
   excluded from comparisons for being trivially perfect. Our D0 at d96/L3
   failing δ=0 is therefore anomalous relative to published behavior.
2. **Two standard protocol elements are missing from our runs: a learning-rate
   sweep and multiple seeds.** Both are load-bearing in this literature —
   published MQAR results show mean-vs-peak accuracy gaps of 35% vs 99% across
   seeds at fixed LR, and a dedicated study (≈3,000 runs) concludes that
   apparent capability failures on MQAR are frequently *optimization* failures.
3. **Our dial task is the pre-MQAR single-query formulation** that Zoology
   explicitly identified as inadequate and moved away from. Multi-query
   (Rung 2) is not an optimization — it is the definitional feature of the
   benchmark.
4. **K3 appears genuinely novel.** The closest published algebraic attention
   (GATr) scores with the *inner product* of multivectors — which our memo §3
   proved is dense-equivalent. No found work uses norm non-multiplicativity or
   zero-divisor structure as the scoring mechanism.
5. **A major mathematical result exists that we did not know about:** the
   sedenion zero-divisor variety is isometric to the exceptional Lie group
   **G₂** (Reggiani 2024). This directly serves the pending X-inequivalence
   certificate and should be read before that work proceeds.

---

## 1. The dial task: MQAR and associative recall

### 1.1 Core references

- Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra, Ré. **"Zoology:
  Measuring and Improving Recall in Efficient Language Models."**
  arXiv:2312.04927 (2023). Introduces MQAR. Code: HazyResearch/zoology.
- Olsson et al. **"In-context Learning and Induction Heads."** (2022). Source
  of the induction-circuit and phase-transition framing.
- Fu et al. **H3** (2023) — earlier synthetic induction-head dataset that MQAR
  supersedes.
- **"Revisiting associative recall in modern recurrent models."**
  arXiv:2508.19029 (2025). ≈3,000 runs, ≈20,000 GPU-hours, dedicated to
  disentangling expressivity from optimization on MQAR and copying.
- Jelassi et al. **"Repeat After Me: Transformers are Better than State Space
  Models at Copying."** (2024). Copying-difficulty dial.
- Hsieh et al. **RULER** (2024). Long-context synthetics incl. multi-key
  needle-in-haystack with controllable distractors.

### 1.2 Findings that bear directly on gate 5

**(a) Attention solves this task class at smaller scale than ours.** Zoology
reports attention solving MQAR perfectly at *all* sequence lengths using a
constant model dimension of **64**, with **two-layer** models. Multiple
follow-up papers exclude softmax-attention transformers from MQAR analysis
entirely on the grounds that they achieve perfect scores across all
configurations. Our D0 is d96 / 4 heads / 3 layers and does not solve δ=0.

*Implication:* capacity is not the binding constraint, and the GPU-dims move
would have been the wrong axis. This is independent confirmation of the
gate-5 handoff's §1 correction, from an external source.

**(b) Learning rate is decisive, and our single LR is a protocol gap.**
Zoology's standard protocol sweeps four learning rates. arXiv:2508.19029's
central thesis is that much of the apparent SSM-vs-attention capability gap is
optimization, not expressivity, and that a suitable learning rate can be
*missed entirely* by a grid — they explicitly compare their wider grid against
Zoology's to demonstrate this. Their result that attention is robust across a
*wide* LR band cuts both ways for us: it argues our failure is not LR, but it
also means the check is cheap and the literature would not accept a single-LR
negative.

**Recommended action: sweep LR ∈ {1e-4, 3e-4, 1e-3, 3e-3} at δ=0 before any
further structural change.** Four CPU cells, same cost as the three we already
ran.

**(c) Seed variance on this task class is extreme.** A 2026 paper reports MQAR
accuracy at kv=16, seq=128, hidden dim 64, 5 seeds: **mean 34.8%, peak 99.3%**
— describing the pattern as the model either partially learning the task or
collapsing entirely, and attributing it to optimization fragility. At kv=32 the
mean collapses to 2.9% with peak still 99.3%.

*Implication:* our single-seed runs cannot distinguish "cannot learn" from
"unlucky seed." **Any conclusion about δ=0 learnability needs ≥3 seeds.** This
is now the second protocol gap, and it is cheap to close.

**(d) Our task is the formulation MQAR was designed to replace.** Zoology's
stated motivation: prior synthetic AR tasks assumed *one query per input at a
fixed position*, with vocabulary smaller than model dimension; real language
requires *multiple recalls at varying positions* from vocabularies *larger*
than model dimension. Our v2 dial is one query, at a fixed final position, with
vocab 129 vs d96 — i.e. squarely the older formulation.

*Implication:* Rung 2 (multi-query) is not a tweak to try if Rung 1 stalls; it
is the standard construction, and adopting it moves the dial task from
"idiosyncratic synthetic" to "recognized benchmark variant," which also
strengthens any eventual write-up. Note the standard MQAR configs use kv ∈
{4, 8, 16, 32} at seq 128 with d=64 — our 16 pairs is mid-range and should be
comfortable.

**(e) Curriculum is used in some setups** (e.g. training from length 128 and
progressing to 256/512/1024 every 20 epochs). Not obviously needed for us, but
noted as an available lever.

### 1.3 What appears novel in our dial

I did not find prior work that holds sequence length, vocabulary, pair count,
and unigram statistics *fixed* while varying only binding demand, with an
empirically verified shortcut floor measured against a suite of non-binding
heuristics. RULER varies distractor counts but does not match statistics that
way. **The near-collision construction with a verified heuristic floor looks
publishable on its own merits**, independent of H4d's outcome — and the v1→v2
leak-and-fix history is itself a methodological contribution (task designs can
contain shortcuts that a single heuristic check misses).

---

## 2. Hypercomplex and algebraic attention (K3's neighborhood)

### 2.1 Core references

- Brehmer, de Haan, Behrends, Cohen. **"Geometric Algebra Transformer" (GATr).**
  arXiv:2305.18415, NeurIPS 2023. Projective geometric algebra G(3,0,1);
  **16-dimensional multivectors**; E(3)-equivariant attention.
- Ruhe et al. **Clifford Group Equivariant Neural Networks**; Brandstetter et
  al. **Clifford Neural Layers for PDE Modeling**, arXiv:2209.04934.
- Tay et al. **Quaternion Transformers / Quaternion multi-head self-attention.**
- Gordeev et al. **"Hypercomplex Transformer: Novel Attention Mechanism."**
  IJCNLP Findings 2025 — generalized hypercomplex attention block, dual-number
  algebra, EEG classification.
- Zhang, Tay, Zhang et al. **"Beyond Fully-Connected Layers with Quaternions:
  Parameterization of Hypercomplex Multiplications with 1/n Parameters."**
  ICLR 2021 — the PHM layer used in V1/V4.
- Mahabadi et al. **Compacter** (arXiv:2106.04647) — low-rank PHM adapters.

### 2.2 The novelty assessment

**GATr is the closest relative and it is meaningfully different.** GATr encodes
tokens as multivectors in a 16-dimensional algebra and builds attention on the
**inner product** of multivectors. Per our kernel memo §3, any real-part /
inner-product bilinear score is exactly a fixed-signature bilinear form and is
absorbed by learned projections — i.e. GATr's attention is (from the
zero-divisor standpoint) dense attention with an equivariance-motivated
structure. That is a perfectly good design for its purpose (E(3) equivariance),
but it means **GATr does not test what K3 tests.**

Likewise quaternion and dual-number attention operate in *division* algebras
(or algebras chosen for computational convenience), where the phenomenon K3
depends on — norm non-multiplicativity — does not exist.

**Conclusion: no found prior work uses norm non-multiplicativity, or
zero-divisor structure, as the attention scoring mechanism.** The K3 design
appears novel. The octonion-degeneracy theorem (memo §2) is, as far as this
search shows, also new — and it is the sharpest available statement of *why*
one must go past the division algebras to get this effect.

Standard caveat: absence of evidence. A targeted follow-up search on
"non-associative algebra attention" and on the Clifford literature's treatment
of degenerate metrics would be worth doing before any paper claim.

---

## 3. Sedenion zero divisors — a major result we did not have

**Reggiani, S. "The geometry of sedenion zero divisors." arXiv:2411.18881
(2024).** Principal result: the set 𝒵(𝕊) ⊂ 𝕊 × 𝕊 of *normalized pairs whose
product is zero* is **isometric to the exceptional Lie group G₂** with a
naturally reductive left-invariant metric. It is further the total space of a
Riemannian submersion over the symmetric space of quaternion subalgebras of the
octonions, with fibers locally isometric to a product of two round 3-spheres.

Supporting literature:
- Moreno, G. **"The zero divisors of the Cayley–Dickson algebras over the real
  numbers."** Bol. Soc. Mat. Mexicana (1998). The foundational structural result.
- Moreno, G. **"Constructing zero divisors in the higher dimensional
  Cayley–Dickson algebras."** arXiv:math/0512517 (2005).
- Biss, Dugger, Isaksen. **"Large annihilators in Cayley–Dickson algebras."**
  Comm. Algebra 36 (2008). [read, 2026-07-20 — abstract + annihilator-scaling
  content used in §3.1(b)]
- Koebisu. **"Determinant Factorization for Left Multiplication in the
  Sedenions."** arXiv:2512.13002 (2025; v1 titled *Singular Structures and
  Geometric Holonomy in the Zero-Divisor Set of the Sedenions*, 15 Dec 2025;
  v2, 26 Mar 2026. Single author, no institutional affiliation). **[read at
  content level, 2026-07-27]** — det L_v = D₁⁴D₂², D₂ = ‖v‖⁴ − 4(‖u‖²‖w‖² −
  ⟨u,w⟩²); the algebraic ZD locus in ℝ¹⁶ is singular, the normalized nonzero
  locus is smooth; Cor. 3.8 gives the classical ZD characterization
  (Re(v₁)=Re(v₂)=0, ‖u‖=‖w‖, ⟨u,w⟩=0), attributed there to Moreno 1998; Thm
  4.2 identifies the image of the first-factor projection of the normalized
  ZD pair variety with the Stiefel manifold V₂(ℝ⁷) (dim 11) — see the
  dimension-14 decomposition in §3.1(d) below. Prior entry here was
  abstract-level only (see §8.4.7 item 4) — this is the correction.
- Biss–Christensen–Dugger–Isaksen. **"Eigentheory of Cayley–Dickson
  algebras."** Forum Math. 21 (2009) 833–851, arXiv:0905.2987. **[read at
  content level, 2026-07-27]** — the full spectral theory this project's
  `cond(L_x)` closed form and annihilator-dimension result rest on; see
  §3.1(a)'s co-citation and §3.1(d) below. Was entirely absent before
  2026-07-27, not merely abstract-level.
- de Marrais, R. Box-kite / ZD-ensemble program, e.g. arXiv:0804.3416.
  [abstract-only]

### 3.1 Why this matters operationally

**(a) It is the X-inequivalence certificate's natural home.** The pending
shuffled-tensor characterization needs an invariant of the null variety.
Reggiani gives the true variety's exact isometry type (G₂, dim 14) and fiber
structure. A shuffled tensor's accidental null set will almost certainly *not*
be a G₂-isometric homogeneous space — which converts the certificate from
"count nulls and hope" into "compare against a known structure." Recommend
Claude Code read this before running item 2 of the v1.0 blockers.

**Addendum, 2026-07-26** (`PHASE5_verification_2026-07-26.md` §2.3): a
cheaper certificate is now the recommended default ahead of the isometry-type
framing above. Moreno (arXiv:q-alg/9710013, 1997) Cor. 1.5 gives
‖xy‖=‖yx‖ for *all* x,y in a Cayley-Dickson algebra of dimension ≥16, not
only zero-divisor pairs — repo-verified: exact to float64 machine precision
(4.4×10⁻¹⁶) for the true tensor S over 5000 random unit pairs, vs. ~15%
median relative asymmetry for the shuffled tensor X at every pinned seed.
Holds on a full-measure set rather than the 14-dimensional zero locus, and
needs no G₂/isometry machinery. The isometry-type framing
(`phase4_reggiani_certificate.py`) remains valid and is not withdrawn, but
the norm identity is the cheaper first check going forward.

**Co-citation, 2026-07-27:** Biss–Christensen–Dugger–Isaksen, *Eigentheory of
Cayley–Dickson algebras* (Forum Math. 21 (2009) 833–851, arXiv:0905.2987)
Lemma 2.10 proves the same identity, ‖xy‖=‖yx‖ for all x,y in any Aₙ, with
**no dimension restriction** (a six-line proof from their Lemma 2.4) — add
alongside Moreno Cor. 1.5's dimension-≥16 statement rather than replacing it.
This paper is also the source of the closed-form `cond(L_x)` used in
`RESULTS_phase4.md` §8.2/§12.1 (Cor. 7.3, Prop. 3.10) and the exact
annihilator-dimension result in (b) below (Prop. 3.20, Prop. 7.4).

**(d) The "dimension 14" figure decomposes — and the total's genericity was
the wrong reason for dismissing it (chat-side finding, 2026-07-27,
repo-verified).** `RESULTS_phase4.md` §8.2 dismisses variety dimension as a
discriminator on the grounds that 14 is generic to any bilinear map
(30 ambient − 16 equations). **The conclusion survives; that reason is
incomplete.** Two different 14s coincide numerically: Reggiani's dim G₂, and
the generic-bilinear-map count. For S specifically the 14 is not an
undifferentiated total — it decomposes as **11 + 3**, forced three
independent ways: Reggiani (𝒵(𝕊) ≅ G₂, dim 14); Koebisu Thm 4.2 (the
first-factor projection's image is V₂(ℝ⁷), dim 2·7−3 = 11, via the
G₂→S⁶→SU(3)→S⁵→SU(2) transitivity chain).

**Flag upgraded, 2026-07-28** (was: "not independently re-derived here,
standard octonion/G₂ Lie theory, plausible but unverified against
source"): fetched Koebisu (arXiv:2512.13002v2, HTML rendition) directly
and read Lemma 3.6 in full, rather than continuing to carry the whole
chain as an unchecked assertion. **Two of the three steps are in the
paper's own proof, verbatim**: "Since `G₂` acts transitively on the unit
sphere in `Im(𝕆)`, there exists `g₁∈G₂` with `g₁u=‖u‖e₁`" (G₂ on S⁶) and
"The stabilizer of `e₁` in `G₂` is isomorphic to `SU(3)`, and its action
on `e₁⊥≅ℝ⁶` is transitive on the unit sphere" (SU(3) on S⁵) — both
sourced, not assumed. **The proof does not address the third step**
(SU(3)'s own stabilizer on that S⁵, i.e. that it is SU(2)) — checked by
direct read, that sentence simply isn't there. That last step is standard
representation theory independent of this paper (SU(n) acts transitively
on S^{2n−1}⊂ℂⁿ with point-stabilizer SU(n−1); n=3 gives stabilizer SU(2)
on S⁵) and is not attributed to Koebisu. **Net: the flag narrows from
"whole chain unverified" to "2 of 3 steps repo-verified against source,
the third is a standard fact outside this paper's scope, correctly
unclaimed by it."**

The third leg of the three-way count stands independently of the above:
BDI Prop. 3.20/7.4 (every eigenspace, hence every annihilator, is a
multiple of 4 — realized exactly at 4, not just bounded by it, confirmed
below). A generic bilinear map's normalized pair variety instead decomposes
as **14 + 0**: full-dimensional base, zero-dimensional (generic,
1-dimensional unnormalized) fiber. **Same total, opposite shape** — S's
zero-divisor locus is codimension 4 in P, not codimension 1, and its
annihilator is 4-dimensional, not 1-dimensional.

**Repo-verified, 2026-07-27:** the annihilator dimension is **exactly 4** at
every one of 20 distinct exact zero-divisor elements checked for S (spectrum
collapses to `{0×4, 1×8, 2×4}` exactly at each, matching BCDI Prop. 7.4
verbatim) — not a bound, realized exactly, every time. **For X, measured at
found near-zero-divisor points (Adam optimization to σ_min ≈ 1e-16, all
three grid seeds): annihilator dimension is exactly 1** — the next-smallest
eigenvalue clears zero by 0.012–0.019 at every seed, no ambiguity. This
matches the "generic bilinear map" prediction exactly and gives a second,
independent binary discriminator (S: 4, X: 1) alongside the eigenvalue
multiplicity signature (S: `(8,4,4)` everywhere; X: 16 distinct everywhere)
— both theorem-backed for S before measurement, both now measured for X, and
both point the same direction: S's zero-divisor structure is a genuine,
non-generic linear phenomenon; X's, where it exists, is generic and
isolated. See `RESULTS_phase4.md` §12.3 for the full record.

**(b) It bears on the ZDTP density question.** My earlier worry — that ZD
density might grow with CD level until the gateway test discriminates nothing —
is exactly the kind of question this literature answers. Biss–Dugger–Isaksen on
"large annihilators" is the direct reference for how annihilator dimension
scales with doubling.

**(c) Sedenion neural networks exist, but not for attention.** Saad Saoud &
Al-Marzouqi, "Metacognitive Sedenion-Valued Neural Network and its Learning
Algorithm," IEEE Access 8 (2020); and sedenion NNs applied to traffic
forecasting (Kopp et al., 2021, Traffic4cast). These use sedenion-valued
weights/activations for compact representation — none use zero-divisor
structure as a mechanism. **Our framing is distinct from the existing sedenion
NN line**, which is worth stating explicitly in any write-up so reviewers do not
mistake it for incremental work on that line.

---

## 4. Attention dials — the β precedent

**Correia, Niculae, Martins. "Adaptively Sparse Transformers." EMNLP 2019,
arXiv:1909.00015.** Replaces softmax with α-entmax, a differentiable
generalization permitting *exactly zero* attention weight, with **α learned per
head**.

Three points of contact with Phase 3:

1. **Structural analogue of β.** A learnable per-head scalar controlling how
   much attention suppresses. Our H3g/H2 nulls should be read against this
   precedent.
2. **Their dial *was* recruited, and differentiated by layer.** They report that
   heads in different layers learn different sparsity preferences, with
   increased head diversity and no accuracy cost. **This is a direct analogue of
   our Phase 3 layer-0 concentration finding** (AIEX-923) — and it suggests the
   layer-wise differentiation we observed is a general property of per-head
   attention dials, not something specific to zero-divisor structure. That
   somewhat *deflates* the layer-0 finding's novelty while *supporting* its
   reality.
3. **Why theirs worked and ours didn't (hypothesis, not established).** α
   modifies the *normalization* — cheap, global, always relevant. β adds a bias
   from a specific feature that must be *useful* to be worth using. This is the
   "load-bearing" distinction in another guise, and α-entmax is a useful
   contrast case for the Phase 3 write-up.

Related: Sukhbaatar et al. (adaptive attention span), Child et al. (sparse
transformers, fixed patterns), Peters/Niculae/Martins (sparse sequence-to-
sequence, the entmax family).

---

## 5. The Phase 3 length-generalization finding

Context for the V1/V4 result (16-way weight sharing → ~3× ppl@1024 improvement,
attributed to sharing rather than algebra):

- Press, Smith, Lewis. **ALiBi** (arXiv:2108.12409, 2021) — establishes that
  sinusoidal/rotary/T5-bias models extrapolate poorly; per-head *slope* is a
  geometric sequence (structural precedent for our forced per-head ladder).
- Kazemnejad et al. (2024) — **NoPE** extrapolates better on downstream tasks.
- Ruoss et al. **"Randomized Positional Encodings Boost Length Generalization."**
  arXiv:2305.16843 — 6,000 models, 15 tasks.
- Zhou et al. **"Transformers Can Achieve Length Generalization But Not
  Robustly."** arXiv:2402.09371.
- Looped/recurrent-depth transformers: weight sharing *across depth* as a route
  to extrapolation.

### 5.1 One caveat that should enter RESULTS_phase3

Zhou et al. report that length generalization is **fragile and significantly
influenced by random weight initialization and training data order, producing
large variances across runs.** Our V1/V4 length-gen result is 3 seeds. The
*direction* is consistent across seeds and the V1≈V4 attribution is safe (both
arms share the confound), but **the magnitude (~3×) should carry a variance
caveat** citing this. Cheap edit, materially improves the claim's defensibility.

### 5.2 Novelty

I did not find a paper specifically claiming *PHM-style weight sharing improves
length generalization*. Weight sharing across *depth* (looped transformers) is
well studied for extrapolation; weight sharing *within* projections via
Kronecker/PHM structure appears studied for parameter efficiency, not
extrapolation. **If that holds under a more targeted search, the Phase 3
exploratory finding — with its shuffled-tensor control showing the effect is
algebra-independent — is a small but genuine contribution.** Worth one focused
search before claiming it.

---

## 6. Recommended actions

**Immediate, gate 5 (supersedes parts of PHASE4_gate5_handoff.md §5):**

1. **Add an LR sweep before anything else.** δ=0, D0, masked loss,
   LR ∈ {1e-4, 3e-4, 1e-3, 3e-3}. Standard protocol in this literature; four
   cheap CPU cells. *This now ranks above Rung 1 (longer run).*
2. **Add seeds.** ≥3 at whichever LR looks best. Published variance on this task
   class (mean 35 / peak 99) makes single-seed conclusions unsafe.
3. **Promote multi-query from Rung 2 to a design correction.** It is the
   definitional feature of MQAR, not an escalation step. Adopting it also aligns
   the dial with a recognized benchmark.
4. **Keep the stop rule.** Nothing here weakens §6 of the gate-5 handoff; if the
   protocol fixes don't produce a learnable δ=0, descoping H4d remains correct.

**Before v1.0 freeze:**

5. Read Reggiani (arXiv:2411.18881) before the X null-variety characterization.
6. Add the Zhou et al. variance caveat to RESULTS_phase3 §4.
7. Cite α-entmax in RESULTS_phase3 §3 as the per-head-dial precedent and as
   context for the layer-0 finding.
8. Position K3 against GATr explicitly in the Phase 4 spec §2 — "closest prior
   work scores with the multivector inner product, which our §3 shows is
   dense-equivalent; K3 scores with the product *norm*, which is where
   zero-divisor structure lives."

**Process:**

9. Adopt prior-art review as a standing gate before pre-registration freeze
   ("after design, before freeze"). This review would have caught items 1–3
   before three cells were spent.

---

## 7. Limitations of this review

Searched 2026-07-20 via web search only; no database or citation-graph
traversal, no MCP academic connector available. Absence of a result here means
"not found," not "does not exist." Three specific gaps I would want closed
before any paper claim:

- a targeted search on non-associative-algebra attention and on degenerate
  metrics in the Clifford literature (§2.2);
- a targeted search on PHM/Kronecker weight sharing vs. length generalization
  (§5.2);
- forward-citation traversal from Zoology and from Reggiani, which is where
  the most recent relevant work will be. **Addressed in §8.3
  (2026-07-23), then corrected in the same section: the Zoology/Reggiani
  traversal itself was under-powered (low a priori information value,
  and blind to anything not citing those specific roots), so §8.3 also
  runs a second, better-rooted pass from GATr/Clifford Neural Layers.
  Not claimed closed — see §8.3's own "left genuinely open" list.**

---

## 8. Follow-up searches (2026-07-21, Claude Code)

Closes the first two of §7's three gaps, at the owner's explicit direction.
Method: web search, several independent query angles per gap, primary
sources fetched (raw HTML/LaTeX where available, not PDF-summarized passes)
for anything that looked load-bearing. Forward-citation traversal (§7's
third gap) is not done here — reading Reggiani itself
(`PHASE4_reggiani_reading.md`, same date) is a down payment on its
Reggiani half, but the traversal proper (checking who has cited Reggiani
or Zoology since) is still open.

### 8.1 Non-associative-algebra attention / degenerate Clifford metrics

**No attention mechanism using octonion/sedenion structure, zero divisors,
or annihilators was found**, across five independent query angles
(octonion/sedenion + attention/transformer directly; "zero divisor" /
"annihilator" + attention; non-associative algebra + ML broadly; Jordan/
alternative algebras + deep learning). This is the same conclusion §2.2
already reached, now with broader query coverage and no change of verdict —
strengthens rather than revises the existing novelty claim.

**One genuinely new finding, from the degenerate-metric side.** Ordinary
(non-degenerate) Clifford algebras of *indefinite* signature already have
zero-divisor-like elements — a null vector $x$ (nonzero, $Q(x)=0$ under an
indefinite but full-rank form, e.g. a lightlike vector in a Minkowski
signature) satisfies $x^2 = Q(x)\cdot 1 = 0$, making $x$ nilpotent and hence
a zero divisor. This is a *different* mechanism from the sedenions'
(indefinite-signature/degenerate-form vs. non-associative Cayley-Dickson
doubling), but structurally adjacent enough to be worth checking whether
anyone attention-scores with it. Two things came out of chasing this:

- The "Metric Learning for Clifford Group Equivariant Neural Networks"
  line (arXiv:2407.09926, building on Ruhe et al.'s CGENN,
  arXiv:2305.11141) discusses degenerate/learnable metrics but — verified
  by direct fetch of the paper text, not just its abstract — **restricts
  to full-rank (non-degenerate) metrics only**, including the Minkowski
  case; null vectors and zero-divisor structure play no functional role,
  and the architecture is equivariant message-passing, not attention.
  Their "degenerate" is about breaking diagonal structure in a learned
  metric matrix, not about a metric with a nontrivial radical. Unrelated
  to what K3 tests.
- **GATr itself — already §2's closest relative — turns out to be a
  literal instance of a degenerate Clifford algebra (projective geometric
  algebra $G(3,0,1)$, where the extra basis vector $e_0$ satisfies
  $e_0^2=0$), and the paper explicitly flags the consequence, verified
  by direct fetch of the paper's own text (not a summary):**
  > "The dot-product attention in Eq. (5) ignores the 8 dimensions
  > involving the basis element $e_0$. These dimensions vary under
  > translations, and thus their straightforward Euclidean inner product
  > violates equivariance."

  i.e. GATr's own authors independently discovered that their inner-product
  attention score is blind to the degenerate direction — structurally the
  same *kind* of blind spot our memo §3 proves for real-part bilinear
  scores generally, but reached from the opposite direction (equivariance
  under translation, not zero-divisor structure). **Their fix is a "join"
  (dualization) operation plus hand-added distance features
  ($\phi(q)\cdot\psi(k)$), not anything that treats the degenerate/null
  direction's algebra as a signal to score with — they patch around it.**
  The paper never uses the words "null," "nilpotent," or "zero divisor."

  **This sharpens, not just repeats, the existing K3-vs-GATr novelty
  claim**: it is not merely that GATr's scoring mechanism happens to be
  dense-equivalent (memo §3, already known); it is that the one place in
  GATr's own construction where degenerate/singular algebraic structure
  is *unavoidably present*, the authors' response was to route around it
  for equivariance reasons, not to ask whether it carries information
  worth scoring with. No found work treats degenerate/zero-divisor
  algebraic structure as the scoring signal itself, under either
  mechanism (indefinite-signature Clifford or Cayley-Dickson doubling).

**Recommended action:** add one sentence to `ZDA_phase4_spec.md` §2's
GATr novelty positioning noting this — a spec edit, held for the owner
per the same discipline as the Reggiani reading, not applied here.

### 8.2 PHM/Kronecker weight sharing vs. length generalization

**No paper was found claiming PHM- or Kronecker-structured *within-
projection* weight sharing improves length generalization**, across five
independent query angles (PHM/Kronecker + length generalization/
extrapolation directly; Kronecker + sequence models + extrapolation;
weight tying within layer + OOD length; structured/Kronecker parameter
sharing + systematic generalization; a direct check of "Higher Order
Transformers With Kronecker-Structured Attention," arXiv:2412.02919 —
verified by fetch to be purely about efficiency for multiway/tensor data,
no length-generalization claim at all).

What the searches consistently surface instead: Kronecker/PHM weight
sharing is well studied for **parameter efficiency** (the PHM paper
itself, Compacter, KronA, MRI-reconstruction and CLIP-compression
variants) and for **systematic/compositional** generalization as a
structural inductive bias — a different axis from length extrapolation.
Weight sharing **across depth** (looped/recurrent-depth transformers,
already cited in §5.1) remains the well-established extrapolation lever;
weight sharing **within a projection's Kronecker factors** (what V1/V4
actually do) has no prior length-generalization claim attached to it in
anything this search turned up.

**This closes §5.2's "worth one focused search before claiming it"
condition — the condition is met, in the negative-evidence sense §5.2
already anticipated.** Standard caveat carried forward: absence of
evidence from web search only, not a database/citation-graph traversal.
The Phase 3 exploratory finding (16-way weight sharing → ~3× ppl@1024
improvement, shuffled-tensor control showing the effect is algebra-
independent) can be written up as a small but apparently genuine
contribution, with this search as the citation-diligence record.

### 8.3 Forward-citation traversal, corrected (2026-07-23)

**First pass (superseded by the second pass below, kept for the
record):** Semantic Scholar Graph API traversal of everyone citing
Zoology (arXiv:2312.04927, 174/174 citing papers scanned by title +
abstract) and Reggiani (arXiv:2411.18881, 1/1 citing paper). Zero
matches for hypercomplex/Clifford/quaternion/octonion/sedenion/zero-
divisor/non-associative-algebra vocabulary in the Zoology set; the one
Reggiani citation was already known (arXiv:2512.13002). **This pass was
methodologically weak and its original write-up overclaimed.** Zoology's
citing community is associative-recall/efficient-attention research —
it has essentially no prior reason to use hypercomplex vocabulary
whether or not a K3-adjacent collision exists elsewhere, so a negative
result there was likely *a priori* and carries little evidential weight
toward a novelty claim. Reggiani's single citation means forward
traversal has no statistical power at all on the sedenion-math branch —
an *absent* instrument, not a clean result. Forward traversal is also
structurally blind to anything that predates or doesn't cite the root
paper — GATr and Clifford Neural Layers (2022–2023, the branch where a
K3-adjacent collision would actually most plausibly live) have no
reason to cite a recall benchmark and would never surface this way.
**Correct, narrow claim this pass supports: nothing in the recall-task
literature descending from Zoology, and nothing citing Reggiani, uses
K3's mechanism.** Not "K3 is novel" — that claim needs the neighborhood
below.

**Second pass, rooted correctly.** Per the owner's direction: forward-
traverse from GATr and Clifford Neural Layers instead of from Zoology,
plus one date-unrestricted keyword pass on "zero divisor" ∧ (neural |
attention | network). Semantic Scholar's API was rate-limited (HTTP 429,
persistent across multiple retries and a ~10-minute window) for the
structured citation-count/list calls this would ideally use, so this
pass is web-search/topic-survey based rather than exhaustive pagination
— weaker coverage than the Zoology/Reggiani pass, acknowledged rather
than concealed:

- **The "geometric algebra attention" family** (found via topic survey,
  not raw citation pagination — 6 papers: GATr itself; CliffordNet, Ji
  2026, arXiv:2601.06793; Haan et al.'s Euclidean/projective/conformal
  algebra comparison, arXiv:2311.04744; Clifford Frame Attention (CFA),
  Wagner et al. 2024, arXiv:2411.05238, extending AlphaFold2's invariant
  point attention with motors in projective GA; GA attention for point
  clouds, Spellings 2021, arXiv:2110.02393; geometric attention for
  many-body systems, Frank et al. 2021, arXiv:2106.02549). **All six
  take the attention score as the scalar (grade-0) projection of the
  geometric or inner product.** That is exactly the real-part-bilinear
  score class our own kernel memo §3 already proves is dense-equivalent
  — the same collision-check GATr alone already passed, now confirmed
  across the whole family this search could find, not just its most
  famous member. None operate beyond Clifford algebras of Euclidean,
  projective (G(3,0,1)), or conformal (G(4,1,0)) signature; none use
  quaternion/octonion/sedenion Cayley-Dickson doubling; none use a null,
  nilpotent, or zero-divisor element as the scoring signal itself — the
  geometric product's structure is consistently used as a feature/message
  representation, never as a deviation-from-multiplicativity score.
- **Direct keyword search, "zero divisor" + neural/attention/network,
  unrestricted by date:** no collision. One tangential, unrelated hit —
  "Geometry of the fibers of the multiplication map of deep linear
  neural networks" (arXiv:2411.19920), on tuples of *matrices*
  multiplying to zero in deep linear nets' parameter geometry, not an
  attention mechanism and not a hypercomplex algebra.
- **Sedenion-valued neural networks, checked because Reggiani's n=1 gave
  the forward-traversal route no power on this branch:** "Metacognitive
  Sedenion-Valued Neural Network" (IEEE, 2020) uses sedenion-valued
  weights for time-series forecasting — parameter-efficiency framing,
  same category as the existing PHM/Compacter references (§2.1), not
  attention, not a zero-divisor-as-signal mechanism. No collision.
- **Left genuinely open, not claimed closed:** STAResNet (arXiv:2408.13619,
  spacetime-algebra ResNet for Maxwell's PDEs) sits structurally closest
  to §8.1's indefinite-signature-Clifford finding (null vectors as
  zero-divisor-like elements) of anything found, but it is a ResNet, not
  attention, and was checked only at the abstract level — whether it
  uses the indefinite signature's null structure *functionally* or just
  as a field representation is unverified. Full structured citation
  counts for GATr and Clifford Neural Layers themselves were not
  obtained (API unavailable this session) — the topic-survey coverage
  above is a substitute, not equivalent to the exhaustive pagination
  done for Zoology/Reggiani, and should be redone via the API when it's
  reachable.

**Framing for any eventual writeup, per the owner's correction: lead
with the positive, structural argument, not the null searches.** The
Cayley-Dickson literature in ML stops at the octonions because division
fails beyond them, and treats the onset of zero divisors at the
sedenions as the reason not to go further (§2's PHM/quaternion
references all sit at or below the octonions for exactly this reason).
K3 inverts that boundary — it uses the annihilation the field has
treated as a stopping condition as the scoring signal itself. That
argument is checkable on its own terms, doesn't depend on search
exhaustiveness, and survives someone later finding an adjacent paper;
the searches in this section are supporting negative evidence, scoped
to the specific neighborhoods they actually covered, not a closure of
the novelty question.

### 8.4 Hypercomplex-ML branch, and the score-vs-projection boundary (2026-07-23)

**Author:** Claude (claude.ai chat), parallel assignment while Claude Code
ran the §8.3 GATr re-root. Delivered as
`PRIOR_ART_REVIEW_zda_section8-4_draft.md`, merged here 2026-07-23 by
Claude Code after independent verification of every load-bearing claim
(method below); content is unchanged from the delivered draft except
this merge note.

**Method and its limits (stated up front, as the original draft did).**
Web search over titles, abstracts, and — where retrievable — full
texts. Weaker than §8.3's Semantic Scholar traversal and weaker than a
structured citation query: topic-directed reading with full-text
confirmation on the one paper that matters, not exhaustive coverage.
Every negative below is absence of evidence. The positive finding in
§8.4.3 is full-text-verified and does not depend on search coverage.

**Verification performed before merge (Claude Code, 2026-07-23):** the
paper anchoring §8.4.3–5 (arXiv:2405.07024, "Demystifying the
Hypercomplex," Comminiello/Grassucci/Mandic/Uncini, IEEE Signal
Processing Magazine) was independently fetched — title/authors/abstract
confirmed, then the specific claims checked against the paper's own
text (not summarized secondhand):
- **Table 1** ("Algebraic convolution and properties for domains")
  confirmed to list Real, Complex, Quaternion, Tessarine, Dual
  Quaternion, Octonion — **no sedenion row**, exactly as claimed.
- §2.1's zero-divisor sentence confirmed verbatim: *"Each time m
  increases, we lose an algebraic symmetry (e.g., commutativity for
  quaternions, associativity for octonions, norm-multiplicativity for
  sedonions)"* and *"all the resulting algebras for m≥4, and
  consequently n≥16, have zero-divisors."*
- §3.3 ("Algebraic Bias") confirmed to derive biases from quaternion
  non-commutativity, octonion non-associativity, and Clifford
  anticommutation, with no mention of zero divisors or annihilation as
  a usable bias.
- §4.1's PHAtt equation confirmed verbatim: **PHAtt =
  softmax(QKᵀ/√d_k)V** — Q/K/V come from PHM layers, but the score
  itself is the ordinary real dot product, exactly as the draft's
  §8.4.4 claims.
- §4.2's equivariance-loss claim confirmed, closely paraphrased in the
  draft: *"PHNNs do not preserve the rotation and translation
  equivariant properties in the 3D space that remain pure biases
  respectively related to quaternion and dual quaternion models."*
- Two Tier-2/3 adjacencies in §8.4.6 spot-checked: arXiv:2606.20547
  ("The Token Is a Group Element," Musialski) confirmed real, with the
  s_ij = −‖log(g_i⁻¹g_j)‖²_λ/τ score formula matching verbatim; its
  claimed explicit positioning against RoPE/LieRE was **not visible**
  in the abstract-level fetch and is unconfirmed (noted, not silently
  passed through). arXiv:2602.10195 ("Versor," Huy & Hirst) confirmed
  real, Cl(4,1)/rotor/WikiText-103 claims match; whether its GATr
  comparison specifically occurs on WikiText-103 (vs. the other
  benchmarks) is unconfirmed at abstract level.
- Not independently re-verified (lower-stakes background citations, not
  load-bearing for §8.4.3/8.4.4's claims): the §8.4.2 Saoud/Bojesomo
  sedenion-network papers and the "[SA20]" Reggiani-citation-resolution
  claim in §8.4.1.

All load-bearing claims held up. Draft content follows unchanged.

#### 8.4.1 Why this branch was invisible to §8.3

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

#### 8.4.2 The branch

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

#### 8.4.3 The positive structural finding (full-text verified)

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

#### 8.4.4 The score-vs-projection boundary — sharper novelty statement

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

#### 8.4.5 Independent support for the V1/V4 outcome (methodology note)

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

#### 8.4.6 Adjacency found outside this branch — Tier 2

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

#### 8.4.7 Open conditions (do not mark §8 closed)

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
4. **Abstract-level entries were being carried as if read (chat-side
   finding, 2026-07-27, accepted).** Koebisu (arXiv:2512.13002) was cited
   in §3 from 2026-07-20 and in §8.3 from 2026-07-23, both times at
   abstract level. Its own Cor. 3.8 was then independently rediscovered
   numerically and briefly written into `PHASE5_PLAN.md`'s planning
   material as a fresh lead before being corrected — the search had
   already succeeded; the citation just hadn't been read past its
   abstract. Distinct from items 1–3 above, which are coverage gaps
   closable by more search: this one isn't, and more keyword queries
   won't fix it. Every §3 entry should be marked **[read]** or
   **[abstract-only]** going forward (done for the entries touched this
   pass; not yet audited for the rest of §3).

#### 8.4.8 Convention flag — **RESOLVED 2026-07-26**

Sources in this sweep report sedenion zero-divisor counts and exemplar
pairs inconsistent with the repo: 84 rather than 336, and pairs such as
(e3+e10)(e6−e15) and (e5+e10)(e6+e9). Almost certainly a combination of
counting convention (ordered vs. unordered, normalized vs. not) and
multiplication-table convention. Per standing rule: flagged, not
reconciled, not averaged. **Any of these figures must be converted into
Baez and into the repo's counting before appearing beside a repo number
in any draft.**

**Resolved by `PHASE5_verification_2026-07-26.md` §1**, run fresh against
`sedenion_kernel.py`: 84 and 336 are both correct, as counts of different
objects in the same lattice — 84 = signed two-blade elements that act as a
left zero-divisor factor (42 index sets × 2 signs), 336 = ordered pairs
(84 elements × 4 partners each, confirmed exactly, no exceptions). A third
object, "elements with unrestricted global sign" or equivalently "unordered
pairs" (336/2, exact because of bilaterality — itself Moreno 1997 Cor. 1.6,
see the same note §2), also comes to 168 — two different objects
coinciding numerically. **This 168 collision is the most likely source of
the cross-source mismatch**: a source quoting "84" or "168" may be counting
any of several distinct objects, not making an error. Convention-mismatch
flag downgraded to a specific, checkable disambiguation; no repo bug or
kernel error found.

#### Net position after §8.3 + §8.4

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
