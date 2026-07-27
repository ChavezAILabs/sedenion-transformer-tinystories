# SESSION_CONTINUATION_HANDOFF — 2026-07-26

**For:** next session — Claude (claude.ai chat) **and** Claude Code
**From:** Claude (claude.ai chat), session of 2026-07-24 → 2026-07-26
**Covers:** ZDA Phase 4 completion, Phase 5 planning, Stage 0; plus IGP24
side work on 2026-07-24.

**Role boundaries (unchanged).** Chat side owns analysis, synthesis, math,
KSJ, and write-ups. Claude Code owns the repo, the validated chain, and all
runs. Chat side proposes repo changes as suggestions for Claude Code to
verify — never as edits applied directly. Owner is the approval gate on all
consequential actions.

---

## 1. Where things stand

**ZDA Phase 4 is COMPLETE at 18/18.** Gate lifted, grand summary read out,
verdicts final. `RESULTS_phase4.md` drafted (chat side) and awaiting Claude
Code verification of every figure against the repo.

**Phase 5 is planned but not started.** No Phase 5 training has been run.
Stage 0 (zero-compute analysis) is 2 of 3 complete.

**IGP24 has a new search direction** from a 2026-07-24 attribution analysis.
Directives written, not yet applied to any engine. Competition closes
**2026-08-15**.

### 1.1 Phase 4 verdicts (final)

| | result |
|---|---|
| H4a | **NEGATIVE** — S − D1 val = +0.1600 vs margin 0.0076 (21× over) |
| H4a-2 | negative — S − D0p ppl@1024 = +6.001 with val parity holding |
| H4b′ | not evaluated (conditional on H4a win) |
| H4c | **PASS** — 6 heads cross in all 3 seeds, 5 of them in layer 0 |
| wall-clock | **EXCEEDED** — S/D0p = 2.39× against a 2× bound |

**One-line summary: the mechanism engages and does not pay.**

### 1.2 Grand summary

| variant | n | val loss (mean ± std) | ppl@512 | ppl@1024 | k1 min | wall h |
|---|---|---|---|---|---|---|
| D0p | 3 | 1.6620 ± 0.0066 | 9.84 | 28.90 | 2.54e+00 | 0.60 |
| D0 | 3 | 1.6400 ± 0.0065 | 7.89 | 20.69 | 2.54e+00 | 0.60 |
| D1 | 3 | 1.6381 ± 0.0042 | 13.34 | 34.04 | 2.53e+00 | 0.67 |
| S | 3 | 1.7980 ± 0.0033 | 11.37 | 34.90 | 1.33e+00 | 1.44 |
| X | 3 | 1.7632 ± 0.0012 | 18.22 | 51.64 | 2.42e+00 | 1.44 |
| Q0 | 3 | 1.8211 ± 0.0079 | 20.54 | 45.88 | 0.00e+00 | 0.54 |

**Primary positive:** the S–X double dissociation, 3/3 seeds, complete
separation on all three axes, single differing variable (the structure
tensor). X's worst val beats S's best; S's worst ppl at both rungs beats X's
best. Lead the write-up with this.

**Normalized extrapolation** (ppl@1024 / e^val, i.e. degradation from
in-distribution): D0 4.01× < D0p 5.48× < **S 5.78×** < D1 6.62× < Q0 7.43×
< X 8.86×. S is third of six. Its extrapolation advantage holds only over X
and D1, not over the simpler dense baselines.

---

## 2. Done this session

### 2.1 Phase 4 completion

- Seed 1339 run to completion in a fresh Colab notebook (`APM_p4_continued`)
  on A100-40GB. All six variants, ~5.3 h.
- The 7/22 partial `D0p_seed1339` (step 16750 of 18311, `ckpt_last.pt`, no
  `summary.json`) was **archived, not deleted**, to
  `MyDrive/p4_runs_archive/D0p_seed1339_partial_20260722`. This was necessary,
  not precautionary: `phase4_grid.py` skips on `summary.json` (line 1085) but
  **resumes on `ckpt_last.pt`** (lines 610–622). Left in place it would have
  silently resumed.
- **Bit-exact reproduction confirmed.** The clean D0p/1339 re-run reproduced
  all eleven eval points from the 7/22 partial identically — train, val, k1,
  every field, four decimals — across a different session, runtime, and
  physical A100. Recorded as a methods finding.
- Data provenance verified: `zda_data_cache` copied to `/content/zda_data`,
  all four files byte-identical by sha256, token counts matching `meta.json`
  (350,000,000 train / 4,884,400 val). `meta.json` present ⇒ line 343 early
  return ⇒ no re-tokenization. Seed 1339 trained on byte-identical tokens to
  1337/1338.

### 2.2 The H4c nan bug (415506e)

The first 18/18 readout printed H4c "negative" with every per-head margin
`nan`. **Root cause:** `h4c_readout()` looped over the `seeds` parameter,
which flows from `main()`'s `--seeds` CLI argument — the same argument
controlling the training loop. The completing invocation passed
`--seeds 1339`, collapsing the grading seed axis to length 1;
`std(ddof=1)` on N=1 divides by zero. Since `nan` is never `<` anything,
every comparison fell through to `False` and "negative" printed **without a
single head being tested.** H4a/H4b′ were unaffected — they auto-discover via
`glob.glob` over `summary.json`.

Fixed at 415506e: seed discovery from disk, refusal to grade on fewer than 3
seeds, and a guard that raises rather than prints on any non-finite margin.
Statistic, margin, and threshold untouched. **Corrected verdict: PASS.**

Corrected readout confirmed H4a bit-identical and the grand summary table
unchanged to every digit.

### 2.3 I1 resolved — amendments are POST-HOC, not blind

The 7/22 partial establishes that seed-1339 data existed **before** the
amendments were drafted (7/23) and frozen (ac2dc3a). Owner was queried and
does not recall whether the partial output was consulted. Per HANDOFF §5
("take the weaker label under any uncertainty"), A1/A2/A3/A3-revised are
labeled **post-hoc**. I1 is closed.

Note: an earlier draft commit message asserted the opposite ("blind property
holds") on the reasoning that no compute occurred *since* drafting. That
premise establishes contamination, not exoneration — compute *after* drafting
preserves blindness, compute *before* forecloses it. Caught before commit.

### 2.4 Artifacts exported — Drive dependency broken

`p4_artifacts/` now in the repo: 76 files, 0.3 MB, all `eval_log.jsonl`,
`summary.json`, `train_log.csv` for all 18 runs plus the archived partial.
**Future readouts run locally; no Colab session required.** Checkpoints
(`ckpt.pt`, ~165 MB × 18 ≈ 3 GB) remain Drive-only.

### 2.5 Phase 5 planned

`PHASE5_PLAN.md` written. Goals: G1 attribution, G2 sparsity falsification
(entmax), G3 width confound (dense at mlp=1824), G4 cost (layer-0 hybrid).
Staged: Stage 0 zero-compute → Stage 1 cheap dense runs → Stage 2 expensive
tensor runs, with a gate between 1 and 2.

**New pre-registration requirement (§7.2): the power check.** Phase 4's H4c
tested *reliability*, not *magnitude* — its bar scaled to each head's own
step-0 variance, so X crossed it in two heads too, and the discriminating
statistic had to be added post-hoc as A1. No Phase 5 criterion may be frozen
without demonstrating it separates the hypotheses it is meant to separate.

**New readout requirements (§7.3):** never emit a verdict from a non-finite
statistic; never share a parameter between an execution loop and a readout
loop; refuse to grade on partial seeds rather than degrading silently.

### 2.6 The bilaterality thread (owner's observation → Moreno)

Owner proposed a two-prong P⊛Q / Q⊛P test. Verified: S is **100% bilateral**
(336/336 ordered pairs, 60/60 general vectors); X is **0%** (median ‖y⊛x‖ ≈
0.93–1.03, indistinguishable from a random pair). Binary discriminator, unlike
variety dimension (14 for both — generic to any bilinear map) or r² magnitude
(continuous, contestable).

Owner then observed the literature counts 84/168, not 336, "with conjugation
symmetry." That pointed at conjugation, and the result is **Moreno 1997**
(arXiv:q-alg/9710013) **Corollary 1.6** — bilaterality holds for all
Cayley-Dickson algebras n ≥ 4, published 29 years ago. The "doubly pure"
strengthening is his **Corollary 1.9**. Reggiani (2411.18881), already in the
prior-art review, cites him.

**Consequence — S_sym was removed as a Phase 5 variant.** Moreno derives
bilaterality from the stronger norm identity ‖xy‖ = ‖yx‖, holding for *all*
x,y (verified: S exact to 4.4e-16; X ~15% median asymmetry at every seed).
Therefore the symmetrized score ‖P⊛Q‖ + ‖Q⊛P‖ = **2‖P⊛Q‖** exactly, and γ is
a learned per-head scalar initialized at 1.0 — so S_sym is S with γ init 2.0.
Training it would measure γ-initialization sensitivity, not algebra. H4a's
analogue H5a was narrowed accordingly, with the power loss flagged explicitly.

**What got stronger:** the norm identity is now the preferred X-inequivalence
certificate — holds everywhere rather than on a measure-zero set, one line to
check, cites to Moreno. `PRIOR_ART_REVIEW_zda.md` §3.1(a) has an addendum
pointing to it; §8.4.8 is marked RESOLVED.

**Counting lattice** (closes 84-vs-336): 42 index sets (= Reggiani's rule
i∈1–7, j∈9–15, i≠j−8, i.e. 7×7−7) × 2 signs = 84 elements × 4 partners = 336
ordered pairs; 168 unordered. **Note 168 denotes two different objects** —
elements-counted-with-sign and unordered pairs — which is the likely source of
cross-source confusion. Three independent confirmations:
`sedenion_kernel.py`, an independent CD construction, Reggiani's index rule.

### 2.7 Stage 0 — 2 of 3 complete

**Item 1, early lock: CONFIRMED (owner's hypothesis).** Using log-slope
regression on layer-0 r²_min (the naive "fraction of descent by eval 2"
metric divides by near-zero for X): S descends throughout at −0.052/eval, all
three seeds negative even excluding the first two evals, min-ever landing late
in 2 of 3 seeds. X moves once by the first eval then **flatlines** at
+0.015/eval, statistically flat.

Reading: a flat-to-positive slope after one move is not what approaching a
barrier looks like — a barrier gives decaying negative slope. Combined with
X's non-monotonicity (min-ever below endpoint in all three seeds), this leans
toward *doesn't* rather than *can't*, without settling it.

**Item 2, cond(L_x) at real init: the conditioning objection is REAL.**
At the actual post-wq/wk + R₈ init distribution, S median ≈ 2.8, X median
≈ 43–44, every seed — within ~1% of the free-sphere prediction.

Note the split: free-sphere **fails** to predict r² (N2: real median ≈0.97 vs
predicted ≈0.04) but **succeeds** for conditioning. Reason: r² is a *joint*
quantity over the q,k pair and the projections correlate them; cond(L_x) is a
*marginal* quantity over single vectors, and the marginals are near-isotropic.
This tells you which free-sphere results transfer.

**Item 3, constrained infimum: BLOCKED on checkpoints.** See §3.1 for the
proposed unblock.

### 2.8 IGP24 (2026-07-24)

See `IGP24_DIRECTIVES_2026-07-24.md`. Headline findings:

- The "Chebyshev octic" seed `y⁸−8y⁶+20y⁴−16y²+1` is **abelian**, not 8T34.
  All roots are 2cos(mπ/24), m coprime to 24; splitting field ℚ(ζ₄₈)⁺, degree
  8, abelian over ℚ. The seed behind the only accepted submission is the
  opposite of the "non-solvable tower" theory that drove ~8 engine streams.
- **The fiber grid is exactly 2-fold redundant:** P₍₋λ,₋c₎(x) = P₍λ,c₎(−x),
  same field, same |disc|. Restricting to λ > 0 doubles coverage free.
- **λ dominates the discriminant by ~125 orders of magnitude.** log₁₀|disc|:
  λ=1 → 45.6–51.2; λ=2 → ~97.2; λ=5 → 168.6–170.4. Since holding a crown
  requires minimum `minimumDiscAbs`, **λ = 1 is essentially forced** (λ=0
  degenerates). Viable grid collapses from ~100 configs to ~12. Spend the
  freed compute on **seed diversity**, not fiber sweeps.
- Of four reported "hits", two were the same polynomial (the λ=±4 pair) and
  both of the others were at ~10¹⁵⁰, unwinnable on discriminant.

---

## 3. Open items

### 3.1 Stage 0 item 3 — split the question, then it unblocks

There are two different infimum questions and **only one needs checkpoints**:

- **Global** — is there *any* (wq, wk) making X's r² small? Answers
  can't-vs-doesn't. **Fresh init with many restarts is the correct method**,
  arguably better than trained weights since restarts explore more.
  **UNBLOCKED — run this first.**
- **Local** — from where training left X, was there a nearby descent
  direction it declined? Answers "was it stuck." Needs the 3 GB sync.

If the global result says X *can* reach low r², the local question becomes
interesting and worth the transfer. If it says X can't, the local question is
moot. Defer the sync until the global result is in.

### 3.2 V3b is back in scope — chat-side correction

When the double dissociation came in, chat side said it "substantially
defeated" the conditioning objection, reasoning that a capacity handicap is
monotone and cannot flip sign between fit and extrapolation.

**That was too strong.** Conditioning is not purely a capacity handicap —
near-singular operators change the *shape* of the attention distribution, and
a sharper-vs-flatter profile can plausibly have opposite-signed effects on
in-distribution fit and on extrapolation. The dissociation rules out simple
severity accounts, not shape-mediated ones. With cond(L_x) now confirmed at
~15× at the real init (§2.7), that alternative is live.

The V3 ladder was moved out of scope because the symmetrized test was going to
answer what V3b was for — and then Moreno killed the symmetrized test. **That
leaves attribution without a control.** V3b (conditioning-matched, algebra
broken) should return to Phase 5 scope. It needs real design time: matching
S's conditioning requires generators that pairwise anticommute, which random
sign flips destroy.

### 3.3 New prior art to check against the review

- **Koebisu, arXiv:2512.13002**, "Determinant Factorization for Left
  Multiplication in the Sedenions." Postdates Reggiani, cites Moreno and
  Reggiani. Gives det L_v = D₁⁴D₂² with D₂ = ‖v‖⁴ − 4(‖u‖²‖w‖² − ⟨u,w⟩²).
- **Biss–Dugger–Isaksen**, cited by Koebisu on the zero-divisor locus.

Both surfaced from a search aimed at something else — a live instance of §8's
own power caveat. §8 remains open per §8.4.7.

### 3.4 Closed-form lead for G4 (speculative)

The ZD locus has an exact closed form, verified to machine precision on 40
variety points: Re(v₁) = Re(v₂) = 0, ‖u‖ = ‖w‖, ⟨u,w⟩ = 0 — four conditions
on 16 numbers, no tensor contraction.

**Caveat:** this characterizes when a *single* element is a zero divisor. The
score r² = ‖P⊛Q‖² is a *pairwise* quantity, so this is not a drop-in
replacement — it's a cheap test for "does P have a good partner," not "is Q
that partner." Whether that's architecturally useful is a design question.
Note under G4; not a directive.

**Addendum (Claude Code, 2026-07-27):** this closed form is Moreno 1998 /
Koebisu (arXiv:2512.13002) Cor. 3.8, not a fresh derivation — both papers
were already in `PRIOR_ART_REVIEW_zda.md` §3 (confirmed at lines
188/190–192/476) since the 2026-07-20 pass, so "verified to machine
precision on 40 variety points" should be read as a kernel-consistency
check against a known result, not as new characterization work. No repo
edit needed beyond this note: `PHASE5_PLAN.md` has no §3.4 and no
closed-form-locus text to correct — the content lives only here.

### 3.5 Carried

| item | notes |
|---|---|
| `RESULTS_phase4.md` figure verification | chat-side draft; verify all numbers |
| Seed-1337 dense figures | chat side never saw those runs |
| §8.4.8 wording | use "84 × 4 partners", not the earlier "2 × 2" |
| S descent range | 77.7–121.3×, not the 99–117× in amendments |
| A3-revised γ dose-response | inference-only, plausibly free tier |
| Paper skeleton | draft before Stage 2, so Phase 5 is designed against a written argument |

---

## 4. Uncommitted work

Nothing from Stage 0 or the verification pass is committed. Awaiting review:

- `PHASE5_verification_2026-07-26.md` (new) — **approved**, commit
- `PHASE5_stage0_findings_2026-07-26.md` (new) — **approved**, commit
- `PHASE5_PLAN.md` — §2 rewritten around Moreno, per-k row deleted, pinned-seed
  numbers substituted, S_sym removed from §4, H5a narrowed in §5, §8 budget
  cut 9→6 Stage-2 runs, §6/§10 checklists updated
- `PRIOR_ART_REVIEW_zda.md` — §8.4.8 RESOLVED, §3.1(a) addendum

**Needs adding before commit:** §3.2 (V3b returns to scope) is a chat-side
correction made after Claude Code's last update and is not yet reflected in
`PHASE5_PLAN.md`.

---

## 5. Compute and budget

- **Measured rate: ≤11.2 units/hr on A100**, verified 2026-07-24, superseding
  the earlier ~13/hr estimate. Update this wherever the old figure is cited.
- Balance was **93.24 units** after D0p/1339; five further variants ran at
  ~4.7 h. **Estimated ~40 units remaining — verify in the Colab UI before
  planning.**
- Stage 1 (D1824, E0; 6 dense runs) ≈ 4 h ≈ **45 units**.
- Stage 2 (X_sym, S_L0; 6 runs after S_sym removal) — re-estimate; was ~215
  units for 9 runs.
- **Keep a reserve.** The Phase 4 grid was interrupted once by unit
  exhaustion and the recovery cost a clean re-run.

---

## 6. Standing orders

**Carried unchanged:** Baez convention; flag convention mismatches, never
average them. Negative results receive the same care as positive ones.
Closed doors stay closed — no K1, no variant R, no β-equivalence test for K3,
`ZD_PAIR` unchanged, γ/β never weight-decayed. Owner is the approval gate.

**Lifted:** the Phase 4 completeness gate (18/18 reached, summary read out).
Grading, verdict language, and `RESULTS_phase4.md` conclusions are now
permitted. Phase 5 drafting is permitted.

**Reopened:** entmax, previously non-graded — that order was conditional on
the completeness gate. It enters Phase 5 as a graded control (E0), not as a
variant of interest.

**New (§7.2/§7.3 of the plan):** the power check before freezing any
criterion; never emit a verdict from a non-finite statistic; never share a
parameter between execution and readout loops.

**Provenance:** `phase4_grid.py` at sha256 `65d419e05b6ec74d…` is the script
that produced all 18 runs; readout corrected at 415506e. **Phase 5 should be a
new script, not an edit**, so Phase 4's provenance stays intact.

---

## 7. Calibration note

Chat side over-claimed **four times** this session, each caught by an external
check rather than by internal reasoning:

1. Claimed the prior-art review missed GATr — it was finding #4 in the
   executive summary, with a stronger argument (dense-equivalence *proved*,
   not merely asserted). Cause: extrapolating from a one-line summary of §8.4
   without reading the document.
2. Claimed X's r² had converged to a floor at ~0.033 on two seeds and
   predicted a third would confirm — seed 1339 came in at 0.02818, 15% below.
   Cause: reading an attractor into n=2.
3. Presented bilaterality as a new theorem — it is Moreno 1997 Cor. 1.6, and
   two citations from material already in the review. Cause: not searching
   before drafting.
4. Said the double dissociation "substantially defeated" the conditioning
   objection — too strong; shape-mediated accounts survive it, and cond(L_x)
   at real init is now confirmed at ~15×. See §3.2.

Also: the per-k shuffle row in `PHASE5_PLAN.md` §2 described a construction
that does not exist in the repo, and the §8.4.8 "2 × 2" decomposition was
asserted rather than measured. Both corrected by Claude Code.

**Both of the session's largest course corrections came from the owner** —
the two-prong bilateral idea, and the observation that the literature counts
84/168 with conjugation symmetry, which led to Moreno and killed a planned
experiment before any compute was spent.

The operating lesson: **verify against an external artifact before a claim
enters the repo.** Claude Code's practice of independently re-deriving rather
than accepting handoff claims caught two chat-side errors this session and
should continue.
