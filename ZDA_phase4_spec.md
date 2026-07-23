# ZDA_phase4_spec — v1.0 (FROZEN 2026-07-22)

Status: **FROZEN — this is the contract.** All §9 blocking items are
closed and every TBD threshold is fixed below. From here the spec may be
changed only by a logged, dated amendment with reason stated (precedent:
`RESULTS_phase2_smoke.md` §6), never revised freely and never
reinterpreted after results exist. Same discipline as
`ZDA_experiment_spec.md`: hypotheses and decision rules fixed before any
run; grading pooled over all seeds; no verdicts other than the
pre-registered ones; negative results written up with the same care as
positive ones.

v0.1 → v0.2 (Claude Code, folding repo-verified derivations): §2
positional scheme settled (R_8(ω_h·pos), uniqueness quantifier stated
precisely); §3 adds D0p and positional-matches both graded dense
baselines; §5 H4a regraded against D0p/D1 with D0 as reference; §8 gates
1–2 marked complete; §9 blocking items 1–2 closed; §10 provenance list
grown to three KSJ captures.

v0.2 → v0.3 (Claude Code, harness-build finding): **variant R (random-
frame K3) dropped as vacuous** — orthogonal conjugation of the
multiplication is exactly absorbed by the learned full-rank projections
(memo §8 addendum, third closed door; verified to 1e-10), and the only
non-vacuous hybrid breaks the positional relative property. H4b′
specificity re-anchored to S vs X (+ Q0 floor + octonion theorem); H4d's
A(δ) re-anchored to D0p − S with S − X attribution at the top level;
dial reduction set now {S, X, D0p}; gate-4 smoke set now S + D0p + X.

v0.3 → v1.0 (Claude, chat session 2026-07-22, owner-approved via
`spec_v1_freeze_diffs.md`): all remaining TBDs frozen — g shape/sign
(§2: s = −γ·r², γ per-head unconstrained, init 1.0), ω_h ladder (§2:
task-pinned geometric, H = 6 worked table), H4c metric/threshold (§5:
per-head p5(r²) on causal support, all-seeds-same-head rule), §6
diagnostic metric fixed to match. Two held addenda applied: Reggiani
smoothness certificate (§9.1) and GATr degenerate-metric positioning
(§2). Entmax-1.5 scoped as instrument arm 2 (§3, non-graded,
non-blocking). Gate 6 closed; grid orchestrator `phase4_grid.py` built
(decision-rule logic executed against synthetic outputs; its header
carries the in-repo launch checklist). §4's dial-family TBDs and §5's
H4d trend-test TBD are moot, not resolved — H4d stands as
pre-registered-but-not-executed, and its text (including its unfrozen
TBD) is retained verbatim per the standing rule; those are the only TBD
markers that survive the freeze, deliberately. Residual factual verifications before grid launch, neither a
threshold: (a) §9.1 certificate figures (σ_min = √2, 3–9%) drafted from
`HANDOFF.md`'s session record — confirm against
`PHASE4_reggiani_reading.md` / `phase4_reggiani_certificate.py` output;
(b) spot-check `phase4_positional.py`'s relative-property at the six
frozen frequencies (frequency-agnostic per head, expected pass). If
either fails: v1.0.1 correction, logged, per the amendment rule above.
**Both verified 2026-07-22 (Claude Code, in-repo): (a)
`phase4_reggiani_certificate.py` rerun live — σ_min(J) = √2 at 336/336
true-tensor null pairs, 0 singularities; X seeds 1337/1338/1339 show
3.1%/7.6%/9.1% singular pairs — matches §9.1 and
`PHASE4_reggiani_reading.md` exactly. (b) rel-prop + shared-phase null
preservation checked at all six frozen ω_h with integer positions to
1023 — worst deviation ~7e-15 (`phase4_ladder_spotcheck.py`, new,
rerunnable). PASS/PASS; no v1.0.1 correction needed.**

Companion documents: `PHASE4_kernel_memo.md` (kernel derivations with §8
repo-verification addendum; all math referenced here),
`verify_phase4_memo.py` / `phase4_positional.py` (rerunnable verification),
`RESULTS_phase3.md` (motivating null and the weight-sharing length-gen
record).

## 1. Question

Phase 3 established that an *optional* zero-divisor gate is sampled by
SGD and priced at perturbative magnitude on an easy task. Phase 4 asks
two things the Phase 3 design could not:

**Q-A (mandatory structure):** When the annihilation manifold is
load-bearing in the score — no parameter setting recovers dense attention
(memo §1, K3) — does zero-divisor structure outperform matched
structure-free and structure-scrambled controls?

**Q-B (task-difficulty moderator — "checkers/chess"):** Does any such
advantage depend on task structural demand? Motivating observation:
shallow tasks create no gradient pressure toward exotic mechanisms;
TinyStories is checkers by construction, so Phase 3's null is conditional
on a checkers-grade task. Phase 3's one real effect (weight-sharing
length-gen) appeared exactly where evaluation exceeded training rehearsal
— weak, indirect, mechanism-mismatched support for the moderator, labeled
as speculation. Phase 4 converts this from post-hoc escape hatch into
pre-registered hypothesis H4d: it gets one shot, with the decision rule
fixed in advance, and if the advantage is absent even at the hard end of
the dial, the escape hatch is closed.

## 2. Primary kernel (fixed)

K3 norm-ratio attention (memo §1): per head, learned projections to full
16-component sedenion coordinates q_i, k_j; score s_ij = g(r_ij²),
r_ij = |q_i ⊗ k_j| / (|q_i||k_j|), Baez convention, Pattern 2 frame
inherited for manifold-referenced diagnostics.

**Novelty positioning (`PRIOR_ART_REVIEW_zda.md` §2, web-search-verified
2026-07-20):** the closest published relative is GATr (Brehmer et al.,
arXiv:2305.18415), which also embeds tokens as 16-dimensional multivectors
but scores attention with the multivector **inner product** — exactly a
real-part bilinear score, which memo §3 (repo-verified) proves is
dense-equivalent, absorbed by learned projections. GATr therefore does not
test what K3 tests: it is a perfectly good E(3)-equivariant design built
on a *different* piece of the algebra than the one K3 depends on. K3
scores with the product's **norm** (r_ij), where zero-divisor structure —
norm non-multiplicativity — actually lives. Quaternion/dual-number
attention operate in algebras where that phenomenon does not exist at all
(division algebras, or algebras chosen for computational convenience). No
found prior work uses norm non-multiplicativity or zero-divisor structure
as the attention scoring mechanism; K3 and the octonion-degeneracy theorem
(memo §2) appear novel, absence-of-evidence caveats noted in the review.
Sharpening the contrast (2026-07-21, `PRIOR_ART_REVIEW_zda.md` §8): GATr
is additionally built on a *degenerate* Clifford algebra, and its own
paper notes its attention score ignores the 8 dimensions involving e₀ —
a deficiency it patches with a dualization trick; i.e. the closest prior
art treats its algebra's null structure as an obstacle to route around,
where K3 makes the analogous structure the scoring signal itself.

- **Shape and sign of g (frozen v1.0):** s = −γ·r², with γ per-head,
  learned, **unconstrained** (no softplus or squaring), init 1.0, never
  weight-decayed (standing invariant). Rationale: (i) the fp dependency
  is closed — §9.2's GPU confirm shows fp32 faithful to ε=1e-5 with a
  live gradient, so no log/sqrt reshaping is needed for numerical
  reasons; (ii) r² is smooth at the annihilation manifold while ∂r/∂q
  diverges as r → 0, so grading H4c's migration story on r² keeps
  gradients well-behaved exactly where they matter; (iii) unconstrained
  γ carries the Phase 1–3 β lesson forward — live gradient at init,
  learned sign as a free diagnostic (§6 iv). The gate-4 smoke ran this
  configuration (γ 1.0 → ~1.1, live), so the frozen choice is
  smoke-tested, not new. A sign-flipped g remains a reparameterization
  of this family, never a separate arm (standing non-variant).
- **Positional encoding (settled — memo §4 boxed notes,
  `phase4_positional.py`):** in-algebra rotation **R_8(ω_h · pos)**, with
  R_8(θ) = cos θ·I + sin θ·L_8 and L_8 = left-multiplication by e8, the
  Cayley–Dickson doubling generator. Uniqueness quantifier, stated
  precisely: among the 15 one-parameter rotation groups generated by
  basis left-multiplications R_i(θ) = cos θ·I + sin θ·L_i (all valid:
  L_i² = −I for all i), **i = 8 is the only generator** for which
  |R(θ)q ⊗ R(φ)k| depends solely on φ − θ (verified on random pairs, tol
  1e-9); no claim is made about orthogonal one-parameter groups outside
  this basis-generated family. R_8 additionally preserves the
  annihilation manifold at all relative offsets — position never moves a
  pair on or off the manifold, so structural nulls are
  position-independent by construction (the §4 feature-vs-confound
  question dissolves). Frequency structure is **forced**: within-head
  multi-frequency (RoPE-style per-plane ladders in L_8's 8 invariant
  planes) breaks the relative property outright (deviation ~3.4), while
  one frequency per head holds it exactly. Therefore each head carries a
  single frequency ω_h and the ladder spans heads; head count bounds the
  positional spectrum. **ω_h schedule (frozen v1.0):** geometric ladder
  pinned to the task rather than to RoPE's base-10000 convention — with
  H heads indexed h = 0…H−1, ω_h = (2π/L_max)^(h/(H−1)), endpoints
  ω_0 = 1 (per-token resolution) and ω_{H−1} = 2π/L_max with
  **L_max = 1024**, the H4a length-generalization eval length, so the
  coarsest head's wavelength covers the longest graded sequence.
  Rationale: the forced design's entire positional budget is n_heads
  frequencies; base 10000 would park the coarsest wavelength at ~63k
  tokens and leave large geometric gaps inside the usable range. The
  **identical ladder, verbatim, goes into D0p and D1**
  (positional-matched by construction), so this choice cannot confound
  any graded comparison; it affects only D0-vs-D0p, which is reported,
  not graded. Worked table at the grid head count H = 6 (Phase 3
  lineage, pinned in `phase4_grid.py`; preflight [P4] there verifies
  the wired ladder against this formula):

  | h | ω_h | wavelength (tokens) |
  |---|---|---|
  | 0 | 1.000000 | 6.28 |
  | 1 | 0.361057 | 17.40 |
  | 2 | 0.130362 | 48.20 |
  | 3 | 0.047068 | 133.49 |
  | 4 | 0.016994 | 369.72 |
  | 5 | 0.006136 | 1024.00 |
- **Architectural consequence, and empirical status (2026-07-20):** under
  the forced design, S's positional resolution scales with head count
  alone (n_heads frequencies total, vs d_head/2 per head for standard
  within-head RoPE) — an inherent cost of the K3-compatible scheme, not a
  choice. Dial-task calibration cell 1 (`PHASE4_three_cells_handoff.md`)
  tested whether this restriction is the binding constraint on
  offset-selective lookup at δ=0 by swapping D0p's per-head ladder for
  D0's full RoPE ladder, everything else fixed: accuracy 0.099 (D0) vs.
  0.125 (D0p) — statistically indistinguishable, both far below
  floor/ceiling and under-trained. **Refuted at these dims: the
  positional ladder is not the binding constraint on H4d's dial task**
  (corrects `PHASE4_three_cells_handoff.md` §6's contingency, which does
  not trigger; see `PHASE4_gate5_handoff.md` §2). §9.3 has the full
  calibration status.

**Explicit non-variants** (doors closed by the memo, listed so they stay
closed): K1 linear collapse score (separable — memo §1); any real-part
bilinear score (identically dense — memo §3); octonion K3 arm (provably
query-independent — memo §2); sign-flipped g (reparameterization);
within-head multi-frequency positional blocks (breaks the relative
property — memo §4 boxed note); random-frame conjugated K3 (absorbed by
learned projections — memo §8 addendum, third closed door; v0.3).

## 3. Variant grid

| id | description | role |
|---|---|---|
| D0p | dense attention, param-matched to S, **positional-matched** (per-head single-frequency RoPE, same across-head ladder as S) | primary baseline (H4a) |
| D0 | dense attention, param-matched, standard within-head RoPE | reference only — prices what the positional restriction costs dense attention |
| D1 | dense attention, FLOP-matched to S, positional-matched as D0p | baseline (H4a) |
| S  | K3-sedenion (true T16), R_8(ω_h·pos) | primary |
| X  | K3, shuffled structure tensor | structure control (H4b′) — V4 analogue; its accidental null set characterized **before** launch (§9.1) |
| Q0 | query-independent attention (key-only scores, matched arch) | no-interaction floor; replaces the octonion arm per memo §2 |

*(v0.3: variant R — random-frame K3 — removed. Orthogonal conjugation of
the multiplication is exactly absorbed by the learned projections
(W → OW), making R the S family under a different init; the only
non-vacuous hybrid breaks the positional relative property. Memo §8
addendum, third closed door. There is no V3 analogue for a mandatory
kernel with learned full-rank projections.)*

Rationale for D0p as primary (v0.2): S's positional scheme is forced to
single-frequency-per-head (§2); grading S against a dense baseline with
the *richer* within-head ladder would confound kernel structure with
positional resolution. D0p isolates the kernel; D0 (reported, not graded)
isolates the positional restriction itself. D0 and D0p share identical
dims, differing only in positional scheme.

3 seeds per variant per task condition. Matching: params ±1% of D0p,
FLOPs ±5% (D1 vs S), with K3's norm/product FLOP accounting audited in
the match table. Identical data order across variants at a seed; same
batch size across all runs (Phase 3 invariants carried forward).

Optional instrument arm (not graded, not counted in the grid): K4
readout with w initialized off e₀, logging w-trajectory toward/away from
the dense-equivalent point (memo §1, K4). Run only if compute allows.

Instrument arm 2 (v1.0, owner-requested, exploratory): **entmax-1.5
normalizer probe.** Motivation: with s = −γ·r² the scores are bounded
above by 0, so softmax can only ever *softly* ignore off-manifold pairs;
entmax-1.5 produces exact zeros — literal selective ignoring, the
behavior the §1 Q-B story says structural demand should reward. Scope,
fixed now:

- α = 1.5 **fixed** (exact sort-based algorithm; no learned α — an
  instrument arm adds no learnable surface).
- Configurations: S and D0p only, **1 seed (1337)**, anchor task only.
  Working ids S-em / D0p-em, outputs under `runs_p4_instrument/`.
- **Not graded, not counted in the grid, cannot trigger or contribute
  to any H4 verdict.** Runs only if compute allows, **after** the graded
  18-run grid completes (same standing as the K4 arm). If the probe
  looks interesting, the only legitimate path forward is a
  pre-registered amendment for a future phase — no post-hoc grading.
- Placement: entmax replaces softmax strictly downstream of the score;
  fp32 score-path hygiene, R_8 positional scheme, structural nulls, and
  the K1 guard (computed on raw score rows, pre-normalizer) are
  untouched by construction. The entmax threshold computation runs in
  fp32, consistent with the score-path rule.
- Chain requirements before any run (no skipping): **vendored** ~30-line
  exact entmax-1.5 (sort-based, Peters et al. 2019) rather than a pip
  dependency, NumPy-mirrored first (`test_phase4_numpy.py` gains a
  P4M8: entmax-1.5 vs mirror on random score rows, incl. rows with ties
  and −inf causal masking), then a torch-side P4T9 (mirror equality,
  causality unchanged, exact-zero support realized, backward finite at
  support boundaries).
- Diagnostics: §6's entropy is logged but **flagged non-comparable**
  across normalizers (different scale; exact zeros). Added, logged per
  head per eval for -em runs only: **support size** (mean and p95
  fraction of nonzero attention weights per row) — the natural sparsity
  trace, and the thing to read against the r² tail traces for any sign
  the exact-zero budget concentrates off-manifold.

## 4. Task axis (H4d) — the difficulty dial

**STATUS 2026-07-21: H4d is pre-registered but not executed** — the dial
task (v2 near-collision and v3 multi-query, protocol-fixed across an LR
sweep and an extended step budget) is not learnable at accessible scale;
full account in `PHASE4_gate5_outcome.md`. Phase 4 proceeds on the anchor
task only (H4a, H4b′, H4c below). Resumption condition, pre-registered: an
H4a surprise positive on the anchor task makes the dial worth resuming
investment in. The rest of this section (§4) is retained as the
pre-registered design it was, not retroactively edited, per the project's
standing rule that negative/undelivered results get the same care as
positive ones.

Two components, both **TBD-concrete at v1.0** but fixed in form here:

- **Anchor task:** TinyStories, unchanged config lineage from Phase 3 —
  the checkers end. Provides direct comparability to the Phase 3 grid and
  the length-gen record (PHM ppl@1024 ≈ 8.2–8.8, the bar any "structural
  ignoring" story must approach to be interesting).
- **Dial task family:** one synthetic family with a scalar difficulty
  parameter δ controlling structural demand, ≥3 levels spanning
  trivial → hard-for-model-scale. Candidates (choose exactly one at
  v1.0, criteria: δ monotonically raises selective-ignoring /
  multi-step-binding demand while vocabulary, length, and token statistics
  stay matched across δ): (a) selective recall with distractor density δ;
  (b) k-hop variable binding with depth δ = k. The chosen family, its δ
  levels, and its eval metric are frozen at v1.0.

Grid cost note: the full design is 6 variants × 3 seeds × (1 anchor +
n_δ dial levels). If compute requires, the pre-registered reduction is:
anchor runs all 6 variants; dial levels run {S, X, D0p} only (H4d needs
exactly these three). Any other reduction requires a spec amendment
before launch.

## 5. Hypotheses and decision rules

All rules graded on pooled mean over 3 seeds; significance margin 2×
pooled std (Phase 3 convention) unless a threshold below says TBD; all
TBD values frozen at v1.0. No grading from partial seeds. No verdicts
outside this list.

- **H4a (value):** On the anchor task, S beats D1 on val loss by more
  than the margin, OR beats D0p's length-gen (ppl@1024) by more than
  margin while within margin of D0p val loss. Otherwise negative.
  D0 is reported alongside as reference (positional-restriction cost)
  but is not a graded comparison.
- **H4b′ (specificity, v0.3):** Conditional on any H4a or H4d win:
  S − X must exceed the margin in the winning metric; else the effect is
  attributed to generic K3-family properties (norm-ratio kernel with
  *some* null structure), not the sedenion zero-divisor geometry
  specifically — noting the §9.1 constraint that X is not null-matched.
  Falsification clause: S ≈ X on all metrics ⇒ nothing
  structure-specific. Q0 is reported alongside as the no-interaction
  floor (descriptive, not a graded comparison). Frame-specificity is
  covered by theorem, not by a run (memo §8, third closed door).
- **H4c (mechanism, pre-loss falsifiable):** During training on any task
  where S wins, per-head distance-to-manifold traces (§6) show
  coordinates migrating toward the annihilation manifold (metric and
  threshold, frozen v1.0: graded on **p5(r²) per head, computed on
  causal support only** — realized pairs; the upper triangle is never
  attended. The tail, not the median, because the gate-4 smoke showed
  medians pinned ~1.0 while the manifold-facing tail did all the moving,
  and a mechanism needs only some pairs near the manifold. H4c passes
  if, on the winning task, **at least one (layer, head)** shows
  end-of-training p5(r²) below its step-0 p5(r²) by more than the
  standing margin — 2× the pooled-across-seeds std of that head's
  step-0 p5 — **and that same head crosses in all 3 seeds**.
  Seed-consistency is the multiple-comparisons control: it admits a
  genuine single-head mechanism without admitting one lucky head in one
  seed. min(r²) stays logged as descriptive only (noisy extreme order
  statistic). Secondary descriptive, not graded: frac r² < 1e-2 lifting
  off zero. Distances are computed against each variant's own null
  structure, per the §9.1 interpretation constraint). If S wins
  a loss comparison but H4c fails, the win is reported as real-but-
  mechanism-unattributed; if H4c fails everywhere, the mechanism story is
  dead independent of loss outcomes.
- **H4d — PRE-REGISTERED BUT NOT EXECUTED (2026-07-21, `PHASE4_gate5_outcome.md`):** the rule below could not be graded because no tested configuration made δ=0 learnable in the first place. Retained verbatim as originally registered.
- **H4d (moderator — checkers/chess, v0.3):** Define A(δ) = [D0p val
  metric − S val metric] at dial level δ (positive = S advantage over
  the positional-matched dense baseline). Prediction: A(δ) is within
  margin of 0 at the lowest δ and exceeds the margin at the highest δ,
  with a monotone trend across levels (trend test TBD at v1.0;
  candidate: sign consistency of A(δ_{i+1}) − A(δ_i) across all steps,
  pooled seeds). Negative if A is flat-null everywhere (closes the
  task-difficulty escape hatch permanently for this kernel family) or if
  A is nonzero but flat in δ (structure helps but difficulty is not the
  moderator — the checkers/chess story specifically is wrong). If A(δ)
  passes, attribution at the top level requires S − X > margin there
  (H4b′ applied at max δ). X at each δ is reported to separate "S gains
  with difficulty" from "any K3 kernel gains with difficulty."

Interpretation table (fixed now to prevent post-hoc narrative drift):
H4a+/H4b′+ = ZD structure adds value; H4a+/H4b′− = K3-family effect,
not ZD; all-null incl. H4d = mandatory formulation also falsified, task
escape hatch closed — program-level stopping evidence for the
"ZD-in-transformers" line absent a new mechanism proposal.

## 6. Mandatory logged diagnostics (the β-trace successors)

Per eval step, per head, all K3 variants: (i) distance-to-manifold
distribution of realized (q,k) pairs (frozen v1.0: r² distribution
summary — min, p5, median pooled across heads, kept for continuity with
the gate-4 smoke logs, **plus per-head p5 and median computed on causal
support only**; the per-head causal-support p5 is the graded H4c input
— since r → 0 *is* manifold proximity); (ii) attention entropy;
(iii) fraction of pairs with
r² < {1e-2, 1e-4}; (iv) γ (or g-parameters) per head. Anchor-task dense
variants log (ii) for comparison. These are burned into the training
script before the smoke run — Phase 3's lesson is that the traces, not
the losses, carried the diagnosis.

## 7. Validation chain

Replacement anchors for T4, per memo §5: structural-null test against the
336-pair table; K1-degeneracy guard (linear readout of the same pipeline
must be query-independent); octonion-degeneracy end-to-end check
(T16 → T8 swap must produce query-independent rows); causality (T6
analogue); NumPy-mirror-first for every kernel component. Numerical
hygiene test for r near 0 under fp16/TF32 before any GPU run. Positional
anchors (new, from §2): R_8 relative-property test and manifold-
invariance test ported into the torch layer tests.

## 8. Sequencing gates (each gates the next; no skipping)

1. ~~Claude Code reproduces memo §7 against the repo kernel.~~
   **DONE 2026-07-19** — `verify_phase4_memo.py`, all checks OK; memo §8
   addendum.
2. ~~Positional derivation (memo §4 open question).~~ **DONE 2026-07-19**
   — `phase4_positional.py`; R_8(ω_h·pos) unique, per-head ladder forced;
   memo §4 boxed notes; folded into §2 here. (g's shape remains TBD
   pending the fp test only — resolved at v1.0, §2.)
3. ~~NumPy mirror + torch layers + §7 invariants green.~~ **DONE
   2026-07-19** — `test_phase4_numpy.py` (P4M1–M7) and `phase4_layers.py`
   (`K3Attention`, tests P4T1–T8: structural nulls exact, K1 guard,
   octonion degeneracy end-to-end, causality, position-shift invariance,
   grads, mirror equality, S/X constructors). fp32 score path per §9.2
   built into the layer. Harness integration (model/config/train loop
   with §6 diagnostics and the answer-position loss mask) is part of
   gate 4 preparation.
4. ~~Trainability smoke (Shakespeare-scale, S + D0p + X, 1 seed).~~
   **PASSED 2026-07-19** (`runs_p4_smoke/`, CPU, 1500 steps, d64/H2/L2):
   no divergence (final val S 2.447, D0p 2.375, X 2.443 from ~4.3);
   K1 guard alive at every eval (min row spread S 0.93, D0p 1.85,
   X 1.66); §6 diagnostics logged and finite — r² distributions centered
   ~1.0 with the manifold-facing tail moving down over training (S min
   r² 0.33 → 0.11; X 0.11 → 0.043; frac r²<1e-2 still 0 — the H4c trace
   works and shows no manifold migration on an easy task, as expected);
   γ live (1.0 → ~1.1 both variants); wall-clock S/D0p = 1.17×
   (X 1.46×), well under the 2× bound. Go/no-go, as fixed above: **GO.**
   (Not graded: dense slightly ahead in-distribution at toy scale,
   consistent with priors.)
5. Dial-task family finalized, δ levels calibrated (D0p must show
   headroom at high δ: not saturated, not floored — calibration runs are
   not graded). **STOPPED 2026-07-21, gate not cleared** — δ=0 was never
   made learnable despite an LR sweep, an extended step budget, and the
   v3 multi-query redesign (`PHASE4_gate5_outcome.md`). Per the
   pre-registered stop rule, H4d is recorded as pre-registered-but-not-
   executed (§4, §5) and Phase 4 proceeds to gate 6 on the anchor task
   only.
6. ~~v1.0 freeze: all TBDs resolved, thresholds fixed.~~ **DONE
   2026-07-22** — γ (§2), ω ladder (§2), H4c (§5), §6 metric frozen;
   held addenda applied (§2, §9.1); entmax instrument arm scoped as
   non-blocking (§3 — it gates nothing, and nothing gates on it beyond
   grid completion). Freeze produced outside the repo from
   `spec_v1_freeze_diffs.md` (owner-approved 2026-07-22); the two
   residual factual verifications are listed in the v1.0 changelog
   note, neither is a threshold. Next: the grid — `phase4_grid.py`
   (built 2026-07-22, decision-rule logic tested against synthetic
   outputs), whose header checklist [1]–[5] (dims tuning, FLOP audit,
   ladder wiring, T=1024 forward, --debug + resume test) gates the
   actual GPU launch.

## 9. Blocking items before v1.0

1. ~~Shuffled-tensor X: characterize accidental null structure per draw;
   fix the draw(s) in the spec.~~ **DONE 2026-07-19** —
   `phase4_shuffledT_nulls.py`. Pinned recipe: numpy port of
   `zda_layers.shuffled_left_mult_matrices` (L_0 = I, L_k random signed
   permutations), draw seed = run seed (1337/1338/1339). Results (vs true
   T16 reference, which reproduces 336/84/σ̃0.481):

   | draw | 2-blade null pairs | null left-dirs | med σ_min(A_x) | dc-RMS |
   |---|---|---|---|---|
   | X 1337 | 96 (0.29×) | 177 | 0.041 | 0.162 |
   | X 1338 | 92 (0.27×) | 177 | 0.040 | 0.159 |
   | X 1339 | 66 (0.20×) | 169 | 0.041 | 0.158 |

   All three draws valid controls: exact nulls exist, interaction
   non-degenerate. **Interpretation constraint, fixed now:** X is not
   null-matched — it has 3–5× fewer exact two-blade null pairs but is
   diffusely near-singular (median σ_min 12× smaller than true T16, and
   generic directions reach σ_min ≈ 1e-5–1e-7). The true tensor's null
   set is sparse and structured; X's is dense and shallow. S-vs-X
   differences must be read against this geometry difference, in either
   direction, and H4c's distance-to-manifold traces are computed against
   each variant's own null structure. Sharper certificate (2026-07-21,
   `phase4_reggiani_certificate.py`, built on Reggiani arXiv:2411.18881):
   the true tensor's exact null pairs sit at exactly σ_min = √2 with
   zero singularities (G₂ acting transitively on the zero-divisor
   variety), while shuffled-tensor draws show scattered σ_min values and
   genuine singularities at 3–9% of their null pairs — S-vs-X
   inequivalence is therefore certified by smoothness/homogeneity of the
   null set, not merely by counts. (Provenance: figures transcribed from
   `HANDOFF.md`'s 2026-07-21 session record; confirm against
   `PHASE4_reggiani_reading.md` / the certificate script's own output
   before grid launch — v1.0 changelog residual (a).)
2. fp16/TF32 near-null behavior test for r — **DONE 2026-07-21** (CPU half
   2026-07-19, GPU half 2026-07-21, both closed). Original CPU sweep
   (informal, not saved as a script): fp32 faithful to ε = 1e-5 with
   correct gradient signs; fp16 storage of q,k quantizes manifold
   distance to an ε ≈ 1e-3 floor (r² reads exactly 0 below; gradient
   dead) and casting up *after* quantization does not help — the loss is
   in input storage, not accumulation; bf16 floor reported as ~1e-2.
   **Hygiene rule, fixed 2026-07-19:** q/k projection outputs and the
   entire score path (product, r², g) are fp32 end-to-end regardless of
   the mixed-precision policy elsewhere; TF32 (same 10-bit mantissa)
   disabled for the product op. Diagnostic bin r² < 1e-4 (§6 iii) is
   meaningful only under this rule.

   **GPU confirm, 2026-07-21** (`phase4_gpu_tf32_confirm.py`, rerunnable,
   same ε-grid/methodology, batched to a realistic attention-score
   shape — this is now the canonical version of the sweep, the original
   was never saved as a script): run on a Tesla T4 (Colab Pro).
   fp32/TF32-off is faithful to ε=1e-5 with a live gradient
   (‖∂r²/∂q‖=4.8e-4 at ε=1e-5) — **the hygiene rule holds on real
   hardware, not just assumed.** fp32/TF32-on gave *bit-identical*
   results to TF32-off at this shape — inconclusive on whether this
   contraction ever routes through TF32 tensor cores on a T4 at Phase 4
   sizes (cuBLAS dispatch is shape/size-dependent); the guard is kept
   regardless, since disabling it costs nothing measurable at these dims
   and removes the ambiguity rather than relying on it staying that way
   at other shapes/GPUs. fp16 floors at ε=1e-4 (exact zero, dead
   gradient) — confirms the CPU finding's mechanism on real hardware,
   floor one order tighter than the original ~1e-3 estimate.
   **Discrepancy, flagged rather than silently corrected:** bf16 showed
   **no floor down to ε=1e-5** on both a CPU rerun and the T4 GPU run
   (same script, same methodology) — contradicts the original informal
   "~1e-2" estimate. The original bf16 number was never captured in a
   saved script, so there's no way to diff methodology against it
   directly; since the new result reproduces identically across two
   independent runs (CPU and GPU) of the same rerunnable script, it
   supersedes the informal estimate. Practical effect: none on the
   hygiene rule itself (fp32 end-to-end is already mandatory regardless
   of what bf16 does), but the earlier "(unusable)" characterization of
   bf16 should not be repeated as established fact.
3. Dial family: **chosen** (near-collision selective recall,
   `PHASE4_dial_task_design.md` + §6 addendum with corrected non-binding
   floor). **δ-level calibration STOPPED 2026-07-21 — gate not cleared,
   H4d pre-registered but not executed** (`PHASE4_gate5_outcome.md`); §4
   and §8.5 have the disposition. This item is item 3 of 3 in this list
   and is the only one that ended un-resolved rather than done — retained
   in full below as the record of what was tried, in the order it was
   tried, per the project's negative-results-get-equal-care rule:
   - **Training objective settled: masked loss (answer-position only) is
     correct and frozen.** A full-next-token-loss alternative was tested
     and refuted in the opposite direction from the original concern —
     it converges exactly to the per-position marginal (val_loss
     4.162–4.163 ≈ ln(64) = 4.1589, accuracy 0.016 ≈ 1/64), i.e. it
     transfers ~zero signal to the answer position regardless of
     positional scheme, rather than relieving any starvation. Masked
     loss is strictly better on every δ=0 run so far and is the
     objective going forward (`phase4_train.py --train_loss masked`, the
     default; `full` is retained as a tested, rejected alternative, not
     removed — no further gradient-instrumentation work planned on it).
   - **Positional ladder refuted as the binding constraint** — see §2
     addendum.
   - **Remaining open question: step budget and protocol.** Masked-loss
     δ=0 runs (D0p 0.125, D0 0.099 acc at 8000 steps) were both still
     descending, not plateaued, when evaluated — loss sits between
     ln(64)=4.159 ("a value token") and ln(16)=2.773 ("one of the 16
     present values"), consistent with a pre-induction phase rather than
     a capacity ceiling (induction circuits typically form via a sharp
     phase transition, not gradual improvement — a hypothesis, not yet a
     conclusion).
   - **External validation, `PRIOR_ART_REVIEW_zda.md` (2026-07-20,
     web-search-verified by Claude Code before acting — see memo/memory
     for the citation checks):** our task class is MQAR
     (Arora et al., arXiv:2312.04927), and standard softmax attention
     solves MQAR **perfectly at d=64, 2 layers, all sequence lengths**
     (confirmed: the first layer performs a shift-by-one combining
     neighboring key/value into one embedding) — many follow-up papers
     exclude attention from MQAR comparisons for this reason. D0 at
     d96/L3 failing δ=0 is therefore anomalous relative to published
     behavior, independently corroborating `PHASE4_gate5_handoff.md`
     §1's correction (capacity is not the constraint). Two standard
     protocol elements are missing from our runs — **a learning-rate
     sweep** (a dedicated ~3,000-run study, arXiv:2508.19029, finds
     apparent MQAR capability failures are frequently optimization
     failures traceable to LR) and **multiple seeds** (published MQAR
     seed variance at comparable scale: mean 34.8%, peak 99.3% over 5
     seeds — single-seed conclusions on this task class are unsafe).
     Further: our v2 dial (one query, fixed final position) is the
     **pre-MQAR single-query formulation that Zoology explicitly
     replaced** — multi-query is the benchmark's definitional feature,
     not an optional escalation.
   - **Escalation ladder, reordered 2026-07-20 by the prior-art
     findings (supersedes `PHASE4_gate5_handoff.md` §5's ordering; its
     stop rule in §6 is unchanged):**
     (1) **LR sweep** — δ=0, D0, masked loss, LR ∈ {1e-4, 3e-4, 1e-3,
     3e-3}, 8000 steps each (four cheap CPU cells); now ranks above the
     longer run below. (2) **D0 masked, 20k+ steps at the original LR**
     — launched 2026-07-20 before the reprioritization, left running to
     completion for the complementary step-budget-at-fixed-LR signal it
     was already 40% through collecting; target is **0.89, not 1.0**
     (the v2 generator's own suffix-heuristic-measured achievable
     ceiling — generic distractors collide with qb at ~1/64 each).
     (3) **≥3 seeds** at whichever LR from (1) looks best. (4)
     **Multi-query episodes promoted from optional escalation to
     required design correction**: 4–8 queries per episode against the
     same 16 pairs, matching the MQAR construction (standard configs use
     kv ∈ {4,8,16,32} at seq 128, d=64 — our 16 pairs is mid-range) —
     requires a `PHASE4_dial_task_design.md`/generator change and
     re-verification against its §2 checks before use for calibration.
     (5) Fewer pairs per episode, recorded as a task-family finding if
     reached, not a plumbing fix. (6) GPU dims, justified on evidence
     only after (1)–(5) — at that point the original three-cells
     decision-table row fires legitimately.
   - **Stop rule (from `PHASE4_gate5_handoff.md` §6, unchanged by the
     prior-art review — see its §6.4): if the protocol fixes above do
     not produce a learnable δ=0, Phase 4 runs on the anchor task only
     (H4a, H4b′, H4c); H4d is recorded in RESULTS_phase4.md as
     pre-registered but not executed, with reason stated (dial task not
     learnable at accessible scale) — not dropped silently and not
     quietly replaced. The checkers/chess insight (AIEX-932, §10)
     survives regardless and remains available for a future experiment
     with a task family that works; if H4a returns a surprise positive,
     H4d becomes worth resuming investment in.
   - **Process note:** prior-art review is adopted as a standing gate
     before pre-registration freeze going forward ("after design, before
     freeze") — this review would have caught the MQAR/LR/multi-query
     gaps before three CPU cells were spent on structural candidates
     instead (`PRIOR_ART_REVIEW_zda.md` §6.9).

(v0.1 items 1–2 — memo repo verification and the positional derivation —
closed 2026-07-19; see §8 gates 1–2.)

## 10. Provenance

KSJ captures, all dated before any Phase 4 number exists:

1. IGP24 lowest-discriminant scoring insight → consolidation logic
   (mandatory integration, fewest-DOF priority), 2026-07-19.
2. Checkers/chess observation + ZDTP Chess cross-reference → H4d
   task-difficulty moderator, 2026-07-19.
3. e8-as-doubling-generator → forced positional design (R_8 uniqueness,
   per-head frequency ladder), 2026-07-19.

Kernel selection and closed doors: `PHASE4_kernel_memo.md`,
independently derived (Claude Desktop) and repo-verified (Claude Code,
memo §8 addendum, 2026-07-19).
