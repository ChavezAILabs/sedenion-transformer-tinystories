# Phase 4 — Gate 5 Handoff: Revised Diagnosis and Stop Rule

**Date:** 2026-07-19
**From:** Claude Desktop (analysis lane)
**To:** Claude Code (repo owner)
**Supersedes:** `PHASE4_three_cells_handoff.md` §5 decision table (see §1)
**Cost of everything recommended here:** ~2 hours CPU + two code reads.
No GPU spend proposed.

---

## 1. Correcting my own decision table

The three-cells handoff said "all three fail → capacity/optimization genuinely
binds → GPU calibration is now the right call." **That row does not fire, and
the fault is in how I wrote it.**

The ≥0.90 criterion was written to prevent partial-credit rationalization on
*converged* runs. Both masked-loss cells were **still descending at step 8000**.
A run truncated before plateau has not failed the bar; it has not yet been
evaluated against it. Applying a convergence criterion to a non-converged
trajectory is a category error, and it is mine.

Corrected reading of the four runs:

| run | variant | train_loss | final acc | final val_loss | state |
|---|---|---|---|---|---|
| ref | D0p | masked | 0.125 | 3.386 | still descending |
| cell 1 | D0 | masked | 0.099 | 3.208 | still descending |
| cell 2 | D0p | full | 0.016 | 4.163 | converged, at optimum |
| cell 3 | D0 | full | 0.016 | 4.162 | converged, at optimum |

**Two of four cells are unfinished, not failed.** The other two are finished and
tell us something definite (§2).

---

## 2. What the cells actually established

**Candidate B (positional ladder) — refuted, cleanly.** Cell 1 (standard RoPE)
matched the D0p reference at 0.099 vs 0.125. Swapping the ladder for full
standard RoPE changed nothing. **This is a genuinely useful result for the
spec:** the per-head frequency ladder is *not* the binding constraint, so H4d's
`D0p − S` comparison is not structurally doomed and §6 of the previous handoff
(grid reshaping, head-count changes, H4d re-scoping) **does not trigger**. My
Candidate B reasoning was wrong and this was the cheapest possible way to learn
it.

**Candidate A (masking starves training) — refuted in the opposite direction,
and Option 3 should be closed.** The full-loss flatline is not a pathology to be
debugged; it is the **correct optimum for that objective**:

- both full-loss runs sit at 4.162–4.163; **ln(64) = 4.1589**
- accuracy 0.016 = **1/64 exactly**
- of 51 positions, ~48 are irreducibly random draws; the answer position is
  1/51 of the objective
- predicting the marginal everywhere *is* the loss minimum, and the marginal at
  the answer position is uniform over the 64 value tokens

The optimizer did its job. **Per-group gradient clipping would be a lossy
re-derivation of masked loss**, which already exists and already does better.
Recommend closing Option 3 rather than instrumenting it.

**Masked loss is the correct objective and should be frozen as such** on the
v1.0 TBD list — decided on evidence, not inherited from my earlier ambiguous
relay wording.

---

## 3. Where the masked runs actually are

Loss 3.21 sits between **ln(64) = 4.159** ("it's a value token") and
**ln(16) = 2.773** ("it's one of the 16 values present in this episode").
Accuracy 0.099–0.125 is roughly 2× uniform-over-present (0.0625).

Diagnosis: the model has learned *the answer is one of the values present in
this episode* but not **which** one. That is the pre-induction phase.

Induction circuits typically form through a **fairly sharp phase transition**
rather than gradual improvement. Being stuck partway with loss still falling is
consistent with "the transition hasn't happened yet," not with "it can't."
This is why step budget is now the leading hypothesis — and note it is a
hypothesis, not a conclusion; see the reliability note in §7.

---

## 4. The cheap check (do this first — code read, no run)

**Confirm each training window holds exactly one complete 51-token episode.**
Context is 64. If the loader packs episodes contiguously into 64-token windows,
most windows contain a truncated episode — pairs with no query, or a query whose
pairs were cut off — and the task is partly unlearnable by construction.

Decode one training batch and verify: `<Q>` marker present at the expected
offset, exactly 16 complete `[ka kb v]` triples before it, answer at the
expected index.

**Also confirm the mask index in y-coordinates.** With `y = x[1:]`, the answer
at `x[50]` lives at `y[49]`. An off-by-one here would be consistent with the
observed plateau.

**Probability this is the cause: low.** The masked runs reach loss 3.21, well
below ln(64), and nothing can beat ln(64) on an irreducibly random target — so
the catastrophic version of this bug is ruled out by the data. But it is a code
read rather than a training run, and it removes the last plumbing doubt before
more compute is spent.

---

## 5. Escalation ladder — cheapest first, each gating the next

**Rung 1 — longer masked run (~1 hr CPU).**
D0, masked, **20,000+ steps**, same config otherwise. Directly tests step budget
as a cause distinct from both refuted candidates. Run well past 8000, since the
event of interest is a phase transition that may not have occurred yet.
*Advance if:* accuracy plateaus below 0.90 with loss also flattening.
*Stop and calibrate if:* accuracy transitions toward ~0.89 (the achievable
ceiling — generic distractors collide with `qb` at ~1/64 each, which is exactly
the 0.894 the suffix heuristic measured; **0.89, not 1.0, is "solved" for
this task**).

**Rung 2 — multi-query episodes (generator change, no extra compute/step).**
Masked loss supplies exactly **one supervised token per episode**, so batch 24 =
24 supervised examples per gradient step. That is a very noisy gradient for a
circuit that must be *discovered* rather than refined. Pack **4–8 queries per
episode** against the same 16 pairs, each contributing a supervised token —
several-fold more signal density at no extra sequence length and no extra
compute per step. This is the standard MQAR-style construction for exactly this
reason.
*Note:* this changes the task definition and must be reflected in the dial
design doc and re-verified against the §2 checks there (fixed length/vocab
across δ, oracle 1.000, all single-token heuristics at floor) **before** it is
used for calibration.

**Rung 3 — fewer pairs per episode.**
If Rungs 1–2 both stall, consider that 16 pairs at 51 tokens may simply be hard
at this scale. Reducing to 8 pairs is a real finding about the task family, not
a plumbing fix, and should be recorded as such.

**Rung 4 — GPU dims.**
Justified on evidence only after 1–3. At that point the original decision-table
row fires legitimately.

---

## 6. Stop rule — the strategic point, stated as a decision not a recommendation

This is the **third consecutive cycle** the dial has absorbed. Meanwhile:

**H4a, H4b′, and H4c are executable today.** Gate 4 passed cleanly, the kernel
trains at 1.17× dense wall-clock, the K1 guard is live, the §6 diagnostics log
and discriminate correctly (the H4c trace already demonstrated it stays quiet on
a task that shouldn't need the manifold). **Only H4d is blocked.**

**Proposed stop rule, to be fixed now rather than after more results:**

> If Rungs 1 and 2 both fail to reach ~0.89 at δ=0, Phase 4 runs on the
> **anchor task only** (H4a, H4b′, H4c), and **H4d is recorded in
> RESULTS_phase4.md as pre-registered but not executed**, with the reason
> (dial task not learnable at accessible scale) stated plainly.

Three points on why this is the honest handling rather than an evasion:

1. **A pre-registered hypothesis that cannot be run is recorded, not dropped
   and not quietly replaced.** "Could not be executed" is a weaker and less
   interesting outcome than a null, and the write-up must not blur them.
2. **The checkers/chess insight survives regardless** — it is captured
   (AIEX-932), dated, and provenance-clean. It remains available for a future
   experiment with a task family that works.
3. **The parent result is expected to null anyway.** Spending further weeks
   making a synthetic task learnable, in service of a moderator hypothesis whose
   parent effect is most likely absent, is poor allocation. If H4a returns a
   surprise positive, H4d becomes worth real investment — and *that* is the
   condition under which the dial work should resume.

---

## 7. Reliability note on this document

I have now been **wrong twice in a row** on this gate: Candidate B (refuted by
cell 1) and Candidate A (refuted by cell 2, in the opposite direction from my
prediction — I had the sign backwards). Both were argued confidently.

The step-budget/signal-density diagnosis in §3 and §5 should therefore be held
**more loosely than its framing suggests**. It is well-motivated and matches
standard practice in the associative-recall literature, but my hit rate on this
gate is 0 for 2.

What is *not* diagnosis and can be relied on: the ln(64) arithmetic in §2 (the
full-loss runs are at their objective's optimum — that is arithmetic, not
inference), the refutation of Candidate B (direct experimental result), and the
0.89 ceiling in §5 (measured by the v2 prototype's suffix heuristic).

Weight §4's checks above §5's theory. They are falsifiable in minutes.

---

## 8. Summary of asks

1. Close Option 3 (full-loss gradient instrumentation) — §2.
2. Freeze masked loss as the training objective on the v1.0 TBD list — §2.
3. Record in spec §2/§6 that the positional ladder is **not** the binding
   constraint at these dims (Candidate B refuted), so the §6 grid-reshaping
   contingency does not trigger — §2.
4. Run the §4 code checks (minutes, no compute).
5. Rung 1: D0 masked, 20k+ steps. Report against the **0.89** ceiling, not 0.90
   — §5.
6. Adopt the §6 stop rule before Rung 2, so the decision to descope H4d is made
   on a pre-set condition rather than by exhaustion.
