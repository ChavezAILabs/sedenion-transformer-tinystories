# Phase 4 — Gate 5 Outcome: Dial Task Not Learnable at Accessible Scale

**Date:** 2026-07-20/21 (autonomous run, owner away; see §6)
**Author:** Claude Code, executing `PHASE4_autonomous_run_plan.md`'s
pre-registered decision tree (§3) literally, branch by branch, as each
stage's result came in.
**Verdict: Stage F fires.** Protocol fixes (LR sweep, extended step budget)
and the task redesign (MQAR-standard multi-query) all failed to reach the
pre-registered bar. Per the plan's stop rule (`PHASE4_gate5_handoff.md` §6,
carried into the autonomous plan §3 Stage F): **H4d is recorded as
pre-registered but not executed.** Phase 4 proceeds on the anchor task only
(H4a, H4b′, H4c), which is ready and unblocked.

---

## 1. Path taken through the decision tree

**A2 → B3 → E2 → F.** Every branch below is graded against the tree's fixed
bar: accuracy ≥ 0.89 at δ=0 (the task's measured achievable ceiling, not
1.0 — generic distractors collide with the query at ~1/64 each even for a
perfect binder).

### Stage A — step-budget probe (D0, masked, v2, δ=0, lr 1e-3)

Target: 20,000 steps. **Interrupted**: the background task returned
`status: killed` at step 18,000/20,000 (90% through), no `summary.json`
written. Investigated before proceeding (plan §0.5): `LastBootUpTime`
unchanged and no sleep event in the System log since before this session,
so this was not an OS-level interruption (no reboot, no sleep); the exact
cause at the background-task/harness level was not further diagnosable
from inside the session and was not chased (plan's "don't invent new
diagnoses" instruction).

Last four evals before the kill:

| step | val_loss | acc |
|---|---|---|
| 9000 | 3.087 | 0.125 |
| 16000 | 2.937 | 0.125 |
| 17000 | 2.869 | 0.156 |
| 18000 | 2.903 | 0.151 |

**Verdict applied: A2**, using the step-18000 data as a stand-in for the
missing final 2000 steps — unambiguous (0.151 acc has no plausible path to
0.89 in 2000 more steps at this rate) and re-running for a qualitatively
identical answer was judged not worth ~2.2h of CPU, per the plan's own
"more-conservative, less-compute" tiebreak (§0.1). Loss was still (slowly,
noisily) descending at the last observed step, not hard-plateaued — noted
per the rule's instruction; the pre-registered no-further-extension
condition still applies regardless.

### Stage B — LR sweep (D0, masked, v2, δ=0, 8000 steps, seed 1337)

| lr | final acc | final val_loss |
|---|---|---|
| 1e-4 | 0.0938 | 3.746 |
| 3e-4 | 0.0990 | 3.409 |
| 1e-3 (reused cell 1, see note) | 0.0990 | 3.208 |
| 3e-3 | 0.0625 | 3.941 |

Reused cell 1's existing lr=1e-3 result (config matches exactly — plan
explicitly permits this) rather than re-running it. **Caught and corrected
a data-integrity near-miss while assembling this table**: the file at
`runs_p4_cells/D0_seed1337_d0/summary.json` (cell 1's original path) had
been silently overwritten by cell 3 (D0+full-loss) before the run-directory
naming fix existed — it currently holds cell 3's number (acc 0.016), not
cell 1's. Used cell 1's number as recorded contemporaneously in the
project's session log/memory (0.0990 / 3.208) instead of the corrupted
file. Fixed prospectively: `phase4_train.py`'s run directories now always
include an `_lr{value:g}` suffix, so this class of collision cannot recur.

**Verdict: B3** — best cell (0.099) is far under the 0.30 threshold, worst
is 0.063. LR is not the binding factor.

### Stage E — v3 multi-query redesign (D0, masked, δ=0, N=16/Q=4, lr 1e-3, 8000 steps)

Ported `dial_v3_prototype.py` → `dial_data_v3.py` (mirror-discipline
lockstep check against the prototype, mask correctness extended to all 4
answer positions per episode, floor-formula match — all pass). Generalized
`phase4_train.py`'s eval accuracy computation from a hardcoded last-position
read to a mask-driven gather over all masked positions (required for v3's 4
positions; verified byte-identical to the old behavior for v2, where the
mask is True only at the last position).

| step | val_loss | acc |
|---|---|---|
| 0 | 5.015 | 0.010 |
| 4000 | 2.948 | 0.090 |
| 6000 | 2.831 | 0.099 |
| 8000 (final) | **2.7805** | **0.0951** |

4× the supervised signal density per episode did not move the needle:
final accuracy (0.0951) is in the same range as every masked-loss v2 run
in this gate (0.083–0.156 across all δ=0 configs tried). **Reminder: v2 and
v3 numbers are not directly comparable** (different generator, different
task) — this is a qualitative, not quantitative, comparison.

**Verdict: E2** — 0.0951 ≪ 0.89. Multi-query does not solve it either.

### Stage C, D — not reached

Both branches require a config that first clears 0.89 at δ=0. None did, so
neither seed-confirmation (C) nor δ-level calibration (D) has anything to
confirm or calibrate. Not executed, not skipped by choice.

---

## 2. What was and wasn't tested

Ruled out, each by direct experiment rather than argument, across this gate
and the two prior handoffs it builds on:

1. **Positional ladder (D0p's one-frequency-per-head)** — refuted
   (`PHASE4_three_cells_handoff.md`, cell 1: D0 with full RoPE performed
   the same as D0p, 0.099 vs 0.125).
2. **Masked training-loss signal starvation** — refuted in the opposite
   direction (`PHASE4_gate5_handoff.md`; full-loss training converges
   exactly to the marginal, ln(64)/1-64, i.e. transfers ~zero signal to the
   answer position; masked loss is strictly better on every run in this
   gate).
3. **Learning rate** (this document, Stage B) — refuted; 4 LRs spanning
   30× all land within a factor of 1.6× of each other, all near-chance.
4. **Task formulation / single-query vs. MQAR-standard multi-query** (this
   document, Stage E) — refuted; the v3 redesign, verified correctly
   ported and wired, does not solve it either.
5. **Plumbing** (`PHASE4_gate5_handoff.md` §4, episode-truncation and
   mask/index alignment) — checked clean by code inspection, not run.

**Not tested, per the plan's explicit non-authorization (§5):** GPU/larger
dims, extending the dial past δ=3 in v3, fewer pairs per episode (plan
Rung/Stage between LR and multi-query in the original gate-5 ladder, folded
into this run as the multi-query stage instead per the prior-art
reprioritization), and any new diagnostic candidate. The honest state is
"four specific, well-motivated causes ruled out," not "capacity is
impossible at any accessible scale."

---

## 3. Scope decision (per the pre-registered stop rule)

Per `PHASE4_gate5_handoff.md` §6 and `PHASE4_autonomous_run_plan.md` §3
Stage F: **Phase 4 proceeds on the anchor task only — H4a, H4b′, H4c —
which is ready and unblocked (Gate 4 passed cleanly 2026-07-19, the K1
guard is live, the §6 diagnostics log and discriminate correctly).**

**H4d is recorded as pre-registered but not executed.** Reason: the dial
task (both the v2 near-collision and v3 multi-query formulations) is not
learnable at the model scale and compute budget accessible to this project,
across the protocol fixes and task redesign tested. This is a weaker and
less interesting outcome than a null result on H4d itself, and is reported
as such rather than blurred into one.

**Resumption condition (pre-registered):** if H4a returns a surprise
positive on the anchor task, H4d becomes worth real investment and the dial
work resumes — at that point with a larger compute budget justified by
having a real effect to characterize, rather than by assumption.

The checkers/chess task-difficulty insight that motivated H4d (AIEX-932)
survives this outcome regardless — it is dated, provenance-clean, and
remains available for a future experiment with a task family that works.

---

## 4. Owed to the project owner (not resolved autonomously, per plan §6)

1. This outcome, with numbers against every rule (above).
2. Reggiani (arXiv:2411.18881, sedenion ZD variety ≅ G₂) has not been read
   for the X-inequivalence certificate — explicitly flagged as **not** to
   start autonomously since it feeds a pre-registration decision
   (`PRIOR_ART_REVIEW_zda.md` §3.1, `ZDA_phase4_spec.md` §9.1).
3. δ-level registration is moot for this outcome (Stage D was never
   reached) — no freeze-time decision pending from this gate.
4. The three targeted follow-up literature searches from
   `PRIOR_ART_REVIEW_zda.md` §7 (non-associative-algebra attention /
   degenerate Clifford metrics; PHM-vs-length-generalization; forward
   citation traversal from Zoology and Reggiani) remain open.
5. Whether to spend GPU time confirming this at larger scale is an open
   question for the owner, not resolved by this document — the plan
   authorized no GPU spend at any branch, so that question was never
   in scope here regardless of outcome.

---

## 5. Provenance

Decision tree: `PHASE4_autonomous_run_plan.md` (Claude Desktop). Prior
corrections it builds on: `PHASE4_three_cells_handoff.md`,
`PHASE4_gate5_handoff.md`, `PRIOR_ART_REVIEW_zda.md` (all Claude Desktop,
2026-07-20). All citation and repo claims in those documents were
independently verified by Claude Code before being acted on (arithmetic,
code inspection, or live web fetch, per document); this outcome document
records only run results produced after that verification, plus the
Stage A interruption investigated directly.
