# Phase 4 Amendment A3 — layer-0 ablation design

**Drafted:** 2026-07-23, with seeds 1337 and 1338 complete, seed 1339
**not yet run**, and the ablation **not yet built**.
**Status: DESIGN FROZEN 2026-07-23 by Claude Code, WITH ONE SUBSTANTIVE
CORRECTION to A3.1's rationale (below) — read that before building
anything.** The intervention itself (γ_h = 0 on targeted heads) is
unchanged and confirmed sound; the *reason given* for why it's the
right intervention was wrong in a way worth fixing before anyone reads
it as design justification. Harness build (A3.2–A3.3) is **not yet
done** — this freeze covers the design document only, per A3.9's own
procedure (steps 1 confirmed; 2–4 remain, deliberately, since building
and validating the harness is separate work, not blocked by this
freeze).
**Author:** Claude (claude.ai chat); corrected/frozen by Claude Code.

---

## A3.0 Purpose

Session-close §3(e) reports the strongest finding of the grid: S drives
min(r²) down 99–117× in L0 where X never moves it, and S's L0 shows
five of six heads with p5 dropping at low γ while deep layers show the
opposite regime in both variants. X has no manifold-seeking early layer.

That is a **correlation**. S both inhabits the low-r² manifold and
extrapolates better than X. Nothing so far connects the two.

§5 item 3 proposes the test: disable S's manifold-seeking L0 heads at
inference and re-measure length generalisation. This amendment fixes the
design before it is run, because an ablation designed after seeing its
own result is worth very little.

**It is designed to be able to fail.** See A3.6 — the negative outcome
is the stronger one, and that asymmetry is registered here in advance so
that a favourable result cannot be over-read later.

## A3.1 The intervention

**Primary intervention: set γ = 0 for the targeted heads at inference.
Nothing else changes.**

**Correction to the rationale below (Claude Code, 2026-07-23, verified
against `phase4_layers.py` directly — `K3Attention.scores()` and
`forward()`):** the analogy to β=0≡standard-attention (T4) is **not
correct for K3**, and `CLAUDE.md`'s own Phase-4 invariants say so
explicitly: *"K3 is mandatory... there is no init-time equivalence to
standard attention to regression-test against... don't try to resurrect
a β=0-style baseline-equivalence test for K3."* Concretely: K3's score
is `s = -γ_h · r²_ij` with **no separate dot-product term** — unlike
Phase 1–3's V2, where β=0 strips an *additive* gate off a standard score
that remains underneath. Setting γ_h=0 does not "leave a content-based
attention pattern intact"; it makes `s ≡ 0` for that head (before the
causal mask), which after softmax is **uniform attention over the
causal window** — content- and position-blind for that head. What
*does* remain intact is the head's own value-vector slice (`wv`'s
per-head split) and the shared output projection `wo`, so the head still
contributes an (unweighted) average of its values rather than nothing —
this is what makes it different from, and more surgical than, deleting
the head, not because content-based routing survives.

Corrected rationale, and why not the obvious alternative:

- γ is per-head (`nn.Parameter` of shape `(n_heads,)`, confirmed by
  direct read) and appears only in the final score formula, so setting
  one head's γ to 0 does not touch `wq`/`wk`/`wv`/`wo` or any other
  head's score — it is exactly as surgical as intended, just not for the
  reason originally given.
- Setting γ=0 removes **the ZD kernel's content/position-selectivity**
  for that head, replacing it with uniform causal-window averaging —
  this is the actual mechanism being removed, and it is a well-defined,
  legitimate ablation target in its own right.
- **Zeroing head outputs would still be the wrong intervention**, for a
  related but distinct reason than originally stated: it would remove
  the head's value contribution entirely (setting its output slice to
  0), which is a strictly larger intervention than γ=0's "average
  instead of select" and would still confound "the mechanism was doing
  work" with "removing a head hurts."
- **Practical consequence for A3.5/A3.6's reading:** the correct
  baseline each targeted head reverts to is *uniform attention*, not
  *standard dot-product attention*. This doesn't change the intervention
  or any pre-specified reading rule below, but any writeup describing
  this ablation must use "uniform attention over the causal window," not
  "standard attention" or "the dot-product term," when characterizing
  what a γ=0 head does.

γ is per-head, so the intervention is per-head and surgical — confirmed,
for the reasons above rather than the β=0 analogy.

## A3.2 Pre-flight harness validation — mandatory, blocking

Before any ablation number is generated, assert on a single batch:

1. **Gate-null identity.** With γ = 0 on the targeted heads, those heads'
   attention logits equal the logits computed with the gate path removed
   entirely, to **1e-5**. **Clarified (Claude Code):** since K3 has no
   term besides `-γ·r²`, "gate path removed entirely" operationally
   means `s := 0` (pre-mask) for that head — this identity is in fact
   *exact* by construction (`s = -0 · r² = 0` for any finite r², and
   `r²` is always finite given the `eps_div` guard), not merely close to
   within 1e-5; the 1e-5 tolerance is a correct-but-conservative choice,
   not evidence the identity is approximate. This checks the harness's
   *wiring* (did it actually zero the right head's γ and nothing else),
   not a deep mathematical claim — see the corrected A3.1 rationale for
   what the resulting attention pattern is (uniform, not "standard").
   This is a wiring analogue of T4, not the same baseline-equivalence
   claim T4 makes for V2 — no dense-attention equivalence is implied.
   If it fails, the harness is wrong and no result from it means
   anything.
2. **Non-interference.** Heads *not* targeted produce bit-identical
   logits to the intact model on the same batch and seed.
3. **Full-ablation sanity.** With γ = 0 on **every** head in every layer,
   S's forward computation has the same functional form as D0p's.
   *This is a form check, not an equivalence claim* — S's weights are
   not D0p's weights, and the two models will not produce the same
   numbers.
4. **Eval determinism.** Eval batches remain seeded by (seed, step) as
   in the grid. The ablation must not consume RNG differently, or
   ablated and intact runs are not comparable.

These four are cheap, run on CPU or any GPU, and can be built and
validated **now**, on existing 1337/1338 checkpoints, without generating
or looking at any ablation result. See A3.7.

## A3.3 Arms

All arms are inference-only, from `ckpt.pt`, on all three seeds.

| arm | model | γ = 0 applied to | purpose |
|---|---|---|---|
| **P** | S | all 6 heads of L0 | primary |
| **C1** | X | all 6 heads of L0 | generic-gate control |
| **C2** | S | 6 heads sampled from L3–L5 | layer-identity control |
| **C3** | S | criterion-selected L0 subset | selection-sensitivity |
| **I** | S, X | none (intact) | baseline, must reproduce grid numbers |

**Arm P targets all six L0 heads, not the five that looked
manifold-seeking.** Selecting five by eye introduces a degree of freedom
chosen after seeing the data. Ablating all of L0 has none.

**C1 is the arm that makes the result interpretable.** X has no
manifold-seeking L0. If X's length-gen degrades as much as S's under the
identical intervention, the effect is generic gate removal and the
manifold-seeking story is unsupported. This arm is not optional.

**C2** must match arm P's head count exactly (6). Head selection within
L3–L5 is by a fixed rule — lowest head index first, two per layer —
fixed here so it is not chosen later.

**C3** uses the criterion *p5(final) < p5(step 0)*, evaluated per seed.
The selected set may differ across seeds; that is acceptable because the
rule is fixed. C3 exists to show whether the result is sensitive to
which L0 heads are targeted. It is secondary and does not carry the
reading in A3.5.

**Arm I must reproduce the recorded grid values** for val, ppl@512 and
ppl@1024 to within floating-point tolerance. If it does not, the
checkpoint-loading path is wrong and everything else is void.

## A3.4 Metrics

Per arm, per seed: **val loss, ppl@512, ppl@1024** — both rungs, per A2.
In-distribution val is included to detect an ablation that simply breaks
the model rather than removing a length-generalisation mechanism.

Primary quantity, the **S–X gap**, per rung r and seed s:

```
G(r, s)  = ppl_X(r, s) − ppl_S(r, s)          [intact,  arm I]
G'(r, s) = ppl_X'(r, s) − ppl_S'(r, s)        [ablated, arms P and C1]
```

Intact values from the two completed seeds, for reference:

| | 1337 | 1338 |
|---|---|---|
| G@512  | 6.53  | 6.83  |
| G@1024 | 12.41 | 17.22 |

**Gap retention:** `R(r, s) = G'(r, s) / G(r, s)`.

## A3.5 Pre-specified readings

Fixed now, blind to every ablation number:

| R | reading |
|---|---|
| R < 0.25 | S–X gap largely eliminated by removing L0 gating |
| R > 0.75 | S–X gap largely preserved; L0 manifold-seeking is incidental to it |
| 0.25 ≤ R ≤ 0.75 | intermediate — report R numerically, **no label** |

Reported per rung and per seed, never averaged across seeds, and stated
as "k of 3" per A1.3.

**Additional pre-specified checks:**

- If **C1 shows X degrading comparably to S** (|ΔS − ΔX| small relative
  to either), the primary reading is void regardless of R — the
  intervention is removing something generic.
- If **C2 (deep layers) degrades as much as P (L0)**, layer identity is
  not doing the work, and any L0-specific claim is unsupported.
- If **in-distribution val degrades sharply** in arm P, the ablation is
  damaging the model broadly and the length-gen change cannot be
  attributed to a length-generalisation mechanism.

Any of these three fires ⇒ report the ablation as uninformative rather
than reaching for an alternative reading.

## A3.6 The confound, and why the negative is the stronger result

**This is an inference-time ablation of a model trained with the gate
present.** The rest of the network adapted to the gate during training.
So a large degradation under ablation conflates two things:

1. the mechanism was doing work at inference (what we want to know), and
2. the rest of the model depends on the gate being present (adaptation
   dependence, which tells us much less).

The clean version is retraining S without L0 gates — one full training
run per seed, ~87 min × 3 on an A100, roughly 57 compute units. That is
out of budget (see A3.7) and is recorded here as the experiment this one
substitutes for.

**Registered asymmetry — the point of this subsection:**

- **R > 0.75 (gap survives) is a strong negative.** If removing the L0
  gating does not close the S–X gap, the manifold-seeking is incidental
  to the length-generalisation advantage. This reading is *not*
  confounded by adaptation dependence, because adaptation dependence
  would predict degradation, not its absence.
- **R < 0.25 (gap dies) is weak and suggestive only.** It is consistent
  with the mechanism mattering, and equally consistent with adaptation
  dependence. It must be written as "consistent with," never as
  "demonstrates," and the retraining experiment must be named as the
  test that would settle it.

Registering this in advance is the whole point: it prevents a favourable
outcome being reported with more force than the design can carry.

## A3.7 Cost and scheduling — this may not need premium units

The ablation is **inference-only**. No training, no optimizer, no
gradient. Cost is a small number of evaluation passes over existing
checkpoints.

Consequences:

1. **Revised budget: well under the ~20 compute units previously
   assumed.** Order 5–15 units depending on eval batch sizing.
2. **It plausibly runs on free-tier hardware.** Short inference jobs fit
   inside free-tier session limits, and no premium GPU is required.
   **If so, the ablation does not compete with seed 1339 for the
   monthly Pro allocation at all.** This should be tested with a single
   arm-I reproduction on a free-tier T4 before assuming it.
3. **Harness build and A3.2 validation can happen now**, during the
   quota wait, on 1337/1338 checkpoints — without generating or
   inspecting any ablation result.

**Scheduling constraint (completeness gate):** build and validate now;
**execute and report only at 18/18**, on all three seeds together. A
two-seed ablation reported early would recreate exactly the
completeness problem the standing orders exist to prevent.

## A3.8 What may not be claimed

- The ablation is **descriptive and post-hoc**. It is not H4a, H4b, or
  H4c, does not bear on them, and its result may not be presented as
  supporting or undermining a graded outcome.
- It addresses the **S-vs-X** comparison only. It says nothing about
  S vs D0p or S vs D1.
- No outcome establishes ZD-specificity on its own. Per the standing
  attribution rule, that requires the controls read together — here,
  C1 in particular.
- A favourable result does not license a Phase 5 spec. Phase 5 remains
  contingent on the grand-summary readout, per standing orders.

## A3.9 Freeze procedure

1. **DONE** — Claude Code confirmed by direct code read
   (`phase4_layers.py`): γ is per-head (`nn.Parameter((n_heads,))`), and
   γ_h=0 nulls that head's score to exactly 0 (not approximately) with
   no other term surviving. **Found and corrected a real error in the
   surrounding rationale** (A3.1 — γ=0 yields uniform attention over the
   causal window, not "content-based attention pattern intact"); the
   intervention itself is unaffected and remains sound.
2. **Not yet done** — harness build and the four A3.2 checks. Deferred
   to the next work session on this thread; not blocking the design
   freeze itself, per the amendment's own separation of "design frozen"
   from "harness validated" (A3.7 item 3 already treats these as
   separate steps: build now, execute/report only at 18/18).
3. **Not yet done** — depends on step 2.
4. **DONE** — file dated and frozen 2026-07-23 (design only, see status
   header). No git repo in this project; the dated header plus
   `HANDOFF.md` log entry is the audit trail in place of a commit hash.
5. Recorded in `HANDOFF.md` in place of a commit hash.

**Ordering caveat — RESOLVED 2026-07-24, LABEL: POST-HOC, NOT BLIND**
(same resolution and correction as A1/A2 — see A1's caveat for the full
reasoning: the 2026-07-22 partial-1339 data predates this amendment's
2026-07-23 draft date, so per the weaker-label rule it's labeled
post-hoc, not blind, regardless of whether it was actually observed).
~~frozen without independent confirmation of seed 1339's run status at
freeze time — flagged to the owner for confirmation.~~ This matters less
for A3 than for A1/A2 anyway, since A3's design doesn't reference
seed-1339-adjacent statistics at all — but the label still applies for
consistency across the series.

**Remaining before execution (unchanged from A3.7's own schedule):**
build the harness, pass all four A3.2 checks, confirm arm I reproduces
grid values — all buildable now, on 1337/1338 checkpoints, without
generating or inspecting any ablation result. Execute and report only at
18/18, per the completeness gate.

---

## Series status

- **A1** — magnitude statistic. Drafted 2026-07-23.
- **A2** — rung posture and comparator discipline. Drafted 2026-07-23.
- **A3** — layer-0 ablation design. This file.

All three are blind to seed 1339 only if committed before it runs.
