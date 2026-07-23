# Phase 4 — Autonomous Run Plan (Gate 5) + v3 Dial Generator

**Date:** 2026-07-20
**From:** Claude Desktop (analysis lane)
**To:** Claude Code (repo owner)
**Purpose:** project owner is away. This document contains a **pre-registered
decision rule at every branch** so no step requires asking. Where a rule is
ambiguous, the default is written out explicitly.
**Attached artifact:** `dial_v3_prototype.py` (verified, checks below).

---

## 0. Standing instructions for autonomous operation

1. **Follow the decision tree in §3 literally.** Every branch has a rule. Do
   not substitute judgment for a written rule; if a result falls between two
   rules, take the *more conservative* branch (the one that spends less
   compute and concludes less).
2. **Do not grade H4a/H4b′/H4c.** Nothing in this plan touches the graded
   experiment. Everything here is ungraded calibration (spec §8.5).
3. **Log everything; conclude nothing.** Record outcomes against the rules.
   The stop-rule decision in §5 is the only place a scope change is
   authorised, and it is already pre-registered.
4. **Never reuse a run directory across configs.** The `_trainfull` fix was one
   instance; the LR sweep and v3 need their own suffixes (`_lr{value}`,
   `_v3`). Colliding directories destroy comparability silently.
5. **If something breaks in a way not covered here, stop and write it up
   rather than improvising a fix.** A halted plan with a clear note is worth
   more than an improvised one.

---

## 1. Why the plan changed (one paragraph)

External prior art (see `PRIOR_ART_REVIEW_zda.md`) establishes that softmax
attention solves the standard benchmark for this task family (MQAR) *perfectly
at model dimension 64 with two layers*, and that published failures on it are
frequently optimization rather than capacity. Two protocol elements standard in
that literature are absent from every run we have done: a **learning-rate
sweep** and **multiple seeds** (published MQAR results at our pair count show
mean 35% vs peak 99% across seeds at fixed LR). Additionally our v2 dial is the
**pre-MQAR single-query formulation** the benchmark was designed to replace.
So: protocol fixes first, then a task redesign, then — only if both fail — the
stop rule.

---

## 2. The v3 dial generator (attached, verified)

### 2.1 What changed and why

**Multi-query.** v2 supplied **one** supervised token per 51-token episode.
Multi-query is the definitional feature of MQAR, not an optimization. v3
supplies **4** (4× signal density at no extra compute per step).

**Grid-block collisions.** The naive multi-query extension of v2 fails
combinatorially: v2 plants collisions *per query*, so Q queries need
`Q(2δ+1) ≤ N`, which at N=16, Q=4 caps δ at 1 — no dial at all. v3 instead
makes collision structure a property of the **pair set**: keys are arranged in
`m × m` blocks (`m = δ+1`), so every key in a block shares its prefix with the
`m−1` others in its row and its suffix with the `m−1` in its column. *Any*
block key is therefore a valid query with exactly the right structure, and Q
decouples from δ. Constraint becomes `B·m² ≤ N` with `B = N // m²` blocks.

**Disjoint token pools.** Blocks and generic distractors draw distinct key
tokens, making the shortcut floor exact rather than approximate. Side benefit
over v2: the queried key's prefix is no longer *specially* repeated — every
block key's prefix repeats equally — so the repetition trace that v2 carried
(δ extra occurrences tied to the answer) is gone.

**δ semantics shift** (must be restated in the design doc): δ is now *block
dimension − 1*, not *number of planted collisions*. Numerically the per-query
collision count is still δ, so the floor formula is unchanged.

### 2.2 Verified properties (3000 episodes/level, N=16, Q=4, δ ∈ {0,1,2,3})

| check | result |
|---|---|
| 1. fixed length / vocab / #answers across δ | 61 tokens, max token 128, 4 answers — **PASS** |
| 2. unigram KL vs δ=0 | 0.0017 / 0.0019 / 0.0021 — **PASS** |
| 3. full-binding oracle | 1.000 at every δ — **PASS** |
| 4. single-token suite vs floor | see below — **PASS** |

| δ | oracle | prefix | suffix | either | floor | floor_either |
|---|---|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1 | 1.000 | 0.506 | 0.509 | 0.347 | 0.508 | 0.344 |
| 2 | 1.000 | 0.338 | 0.341 | 0.211 | 0.344 | 0.213 |
| 3 | 1.000 | 0.256 | 0.255 | 0.147 | 0.262 | 0.156 |

Floor formula **unchanged from v2**: `floor(δ) = 1/(δ+1) + (1−1/(δ+1))/n_val`.

**Sequence length 61 fits the existing ctx 64** — no config change needed.
Extending the dial past δ=3 requires N=25 (m=5), which needs seq 88 and
therefore a ctx increase; **do not do this without an explicit decision.**

### 2.3 Port requirements

- Reproduce all four checks in the repo copy before any training run.
- Answer mask is now **4 positions per episode** (returned as `ans_pos`), not 1.
  The existing mask-correctness test must be extended: loss over non-answer
  positions excluded exactly, at all four positions.
- Accuracy is now over 4 supervised tokens per episode.
- Keep the Phase 3 seeding invariant: identical episodes across variants at a
  seed; eval seeded by (seed, step, δ).
- **v2 and v3 numbers are not comparable.** Do not plot them on one axis.

---

## 3. Decision tree — execute in order

Success bar throughout: **accuracy ≥ 0.89 at δ=0** (the task's measured
achievable ceiling, not 1.0). All δ=0 runs use D0 + masked loss unless stated.

### Stage A — Rung 1 (already running, ~80 min remaining)
D0, masked, v2 generator, δ=0, 20,000 steps, lr 1e-3, seed 1337.

- **A1 — final acc ≥ 0.89** → step budget was the answer. Skip to **Stage C**
  (seed confirmation at lr 1e-3, 20k steps).
- **A2 — final acc < 0.89** → proceed to **Stage B**. Record whether loss was
  still descending at 20k; if it was, note it, but *do not extend again* — the
  pre-registered kill condition in the gate-5 handoff applies.

### Stage B — LR sweep (4 cells, ~3 h)
D0, masked, **v2 generator** (keeps it a clean single-variable change from the
runs in hand), δ=0, 8,000 steps, seed 1337,
**lr ∈ {1e-4, 3e-4, 1e-3, 3e-3}**. Run dirs suffixed `_lr{value}`.
(The lr 1e-3 cell already exists as cell 1 — rerun anyway for identical step
count, or reuse if config matches exactly.)

- **B1 — any cell ≥ 0.89** → LR was the answer. Go to **Stage C** at that LR.
- **B2 — best cell in [0.30, 0.89)** → partial. Go to **Stage C** at that LR
  with 20,000 steps.
- **B3 — all cells < 0.30** → LR is not the binding factor. Go to **Stage E**.

### Stage C — seed confirmation (3 cells)
Same config as whichever branch sent you here, seeds **1337, 1338, 1339**.

- **C1 — ≥ 2 of 3 reach ≥ 0.89** → δ=0 is solved. Go to **Stage D**.
- **C2 — otherwise** → go to **Stage E**. (Note in the log: high seed variance
  on this task class is *expected* per prior art, so 1-of-3 is not a solve.)

### Stage D — δ-calibration (the actual gate-5 deliverable)
Whichever generator solved δ=0, at the solving LR/steps/seeds. D0p at each
available δ level (v2: {0,2,4,7}; v3: {0,1,2,3}).

- Pick as **registered dial levels** those where D0p sits **clearly above
  floor(δ) and clearly below 0.89** — that is headroom.
- Require **≥ 3 levels including δ=0**. If fewer than 3 qualify, report the
  curve and stop; level selection then needs an explicit decision.
- Calibration is **ungraded**. Do not freeze the spec; report the curve.

### Stage E — v3 multi-query redesign
Port `dial_v3_prototype.py` per §2.3, reproduce the four checks, then run
D0 + masked, δ=0, N=16, Q=4, at the **best LR found in Stage B** (default
3e-4 if Stage B was skipped), 8,000 steps, seed 1337.

- **E1 — acc ≥ 0.89** → go to **Stage C** (seeds), then **Stage D**.
- **E2 — acc < 0.89** → go to **Stage F**.

### Stage F — stop rule (pre-registered, gate-5 handoff §6)
Both protocol fixes and the task redesign have failed. **Do not escalate to
GPU dims. Do not generate new candidate diagnoses.**

Action: write `PHASE4_gate5_outcome.md` recording every stage's numbers
against its rule, and update the spec to record **H4d as pre-registered but
not executed**, reason: *dial task not learnable at accessible scale*. Phase 4
then proceeds on the **anchor task only** (H4a, H4b′, H4c), which is ready and
unblocked. Resumption condition, also pre-registered: an H4a surprise positive
would make the dial worth real investment.

---

## 4. Time and compute budget

| stage | cells | est. wall |
|---|---|---|
| A (running) | 1 | ~80 min remaining |
| B | 4 | ~3 h |
| C | 3 | ~2.5 h |
| D | 3–4 | ~3 h |
| E | 1 (+ port) | ~1 h |

Worst path (A2 → B3 → E2 → F) ≈ 5 h of CPU. Best path (A1 → C1 → D) ≈ 6 h.
All CPU, no GPU spend authorised by this document.

---

## 5. What is explicitly NOT authorised here

- GPU / Colab calibration runs.
- Extending the dial past δ=3 in v3 (needs ctx increase).
- Any change to the graded experiment (H4a/H4b′/H4c) or the anchor task.
- Freezing the spec to v1.0.
- Further step-budget extensions beyond Stage A.
- New diagnostic candidates if Stage F is reached.

---

## 6. Open items owed to the project owner on return

1. Outcome of the decision tree, with numbers against each rule.
2. Reggiani (arXiv:2411.18881, sedenion ZD variety ≅ G₂) has not yet been read
   for the X-inequivalence certificate — flagged in the prior-art review §3.1,
   still pending, and **should not** be started autonomously since it feeds a
   pre-registration decision.
3. If Stage D produced a headroom curve, the registered δ levels are a
   freeze-time decision, not an autonomous one.
4. The three targeted follow-up searches in prior-art review §7 remain open.

---

## 7. Reliability note

The analysis lane went 0-for-2 on diagnoses at this gate (positional ladder;
loss masking — the latter wrong in the opposite direction). The current plan is
**not** a third diagnosis: Stages B and C are standard protocol from the
benchmark literature, and Stage E is adopting the benchmark's own task
formulation. That is why the plan leads with them rather than with a new
theory. Stage F exists because the possibility that none of this works is real
and was pre-registered before these runs, not after.
