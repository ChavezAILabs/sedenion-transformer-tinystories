# Phase 4 — Three Cheap Cells Before GPU Calibration

**Date:** 2026-07-19
**From:** Claude Desktop (analysis lane)
**To:** Claude Code (repo owner)
**Status:** recommendation — GPU calibration is **no-go** until these read out.
**Cost:** ~45 min background CPU each, ~2¼ hours total, laptop only.

---

## 1. Why this document exists

The overnight chain returned D0p at δ=0 with final accuracy **0.125**, and the
conclusion recorded was "capacity precondition fails decisively at smoke dims →
move δ-calibration to GPU dims." I think that conclusion is premature, and the
one intervention it prescribes (bigger dims) is the one that would *not* fix
either of the two most likely causes.

**The core objection:** at δ=0 the dial task reduces to induction/lookup —
locate the token pair, copy the value at a fixed offset. That is the most
reliably-learned circuit in small transformers, forming in far smaller models
than d96/H4/L3 with 8000 steps. When a model fails the easiest known circuit at
several times sufficient width, "insufficient capacity" is rarely the
diagnosis; something upstream is usually preventing the circuit from forming.

Two candidates, both untested, both cheap to isolate. Neither is addressed by
scaling dimensions.

---

## 2. Candidate A — answer-only training loss (signal starvation)

**Status:** flagged in the dial design doc relay list, never given a
discriminating run.

If the *training* objective is masked to the answer position, each 51-token
episode contributes exactly one gradient-bearing token — roughly 2% signal
density. The model must discover the `[label label value]` triple structure,
the value-token subspace, and the copy operation from that single token's
gradient, with no gradient anywhere else to scaffold them.

For reference, MQAR-style associative-recall setups that train on answer
positions generally pack **many** query–answer pairs per sequence, precisely to
avoid this. One query per 51 tokens is very sparse by comparison.

**The observed number fits this story.** 0.125 at δ=0 is ~8× uniform over the
64-token value vocabulary and ~2× uniform over the 16 values actually present
in the episode. So the model has learned the output-space restriction and some
copying bias, but not the lookup. That is the shape of signal starvation, not
of insufficient width.

**Origin note, for the record:** the ambiguity is mine. The dial design doc
relay item said "answer-position loss requires a loss mask in the training/eval
loop," written to make the *graded metric* computable but readable as "mask the
training objective too." Whatever the harness currently does, the training
objective is at present an accident of that sentence rather than a decision,
and it needs to become a decision on the v1.0 freeze list — it is harness-level
and shared across all variants.

**Proposed resolution if A is confirmed:** train with full next-token loss over
the whole sequence (incompressible positions contribute noise gradient; the
structural positions carry real signal), and **grade** on answer-position loss
and answer-token accuracy at eval. Graded metric unchanged; training signal
restored.

---

## 3. Candidate B — D0p is the wrong probe for a positional task

**Status:** structural, follows directly from the forced positional design.

The Phase 4 positional derivation forces **one frequency per head**
(`R_8(ω_h·pos)`, ladder across heads). At d96/H4, standard RoPE gives each head
d_head/2 = 12 frequency components; D0p gives each head exactly **one**.

A head with a single frequency has positional dependence that is a single
sinusoid in (i − j). It cannot build a sharply offset-selective kernel — and
"locate this pair, copy at a fixed offset" is exactly an offset-selective
operation. So the capacity probe was run on the positionally handicapped
baseline, on a task that is fundamentally positional lookup. That conflates
*"task too hard at these dims"* with *"task too hard with one frequency per
head."*

The natural capacity probe is **D0** (standard RoPE) — the strongest dense
model, and already in the grid as the ungraded reference that "prices the
positional restriction." It just needs running **before** the freeze rather
than after.

---

## 4. The three cells

All at the current overnight config (**d96/H4/L3/mlp384, batch 24, 8000 steps,
lr 1e-3**), v2 balanced-collision generator, **δ = 0**, seed 1337, CPU
background. Changing only one factor per cell against the run already in hand.

| cell | model | training loss | purpose |
|---|---|---|---|
| — | D0p | answer-masked | **already have it: 0.125** (reference) |
| **1** | **D0** (standard RoPE) | answer-masked | isolates Candidate B |
| **2** | **D0p** (per-head ladder) | full next-token | isolates Candidate A |
| **3** | **D0** (standard RoPE) | full next-token | run only if 1 and 2 both fail |

Cell 3 is conditional: if either 1 or 2 solves δ=0, the responsible factor is
identified and cell 3 is unnecessary. If both fail, cell 3 tests whether the
factors are jointly necessary before anything is concluded about dims.

**Graded output per cell:** answer-token accuracy at eval (primary), plus
answer-position loss. Everything else (seed discipline, eval seeding by
(seed, step, δ), identical episodes across cells) per the existing Phase 3
invariants.

**Success criterion, fixed now:** δ=0 is "solved" at **accuracy ≥ 0.90**. It is
an unambiguous lookup with a 1.000 oracle; anything materially below 0.90 means
the circuit did not form. Partial credit is not informative here.

---

## 5. Reading the outcomes

| result | diagnosis | next step |
|---|---|---|
| Cell 1 solves, cell 2 doesn't | **positional ladder is binding** | see §6 — this reshapes the grid |
| Cell 2 solves, cell 1 doesn't | **loss masking was starving training** | switch training objective, freeze it, re-probe δ=7, then calibrate |
| Both solve | both factors independently sufficient | adopt full-loss training **and** record the ladder cost; re-probe δ=7 |
| Both fail, cell 3 solves | factors jointly necessary | adopt both; re-probe δ=7 |
| All three fail | capacity/optimization genuinely binds at smoke dims | **now** GPU calibration is the right call, on evidence rather than assumption |

Note the last row: this exercise is not an argument against GPU calibration.
It is an argument for spending that budget only after the cheap alternatives
are excluded — and if all three fail, the GPU move is *better* justified than
it is today.

---

## 6. Why this is a spec matter, not just debugging

If cell 1 solves δ=0 and cell 2 does not, the per-head frequency ladder is
itself costly on recall-type tasks. **S inherits the identical restriction** —
it is forced by the positional derivation, not a choice. In that case H4d as
currently written would compare two positionally-crippled models (S and D0p) on
a task neither can perform, and A(δ) = D0p − S would be a difference of noise.
The dial would measure nothing, and a null would be uninterpretable.

Options in that event, none of which should be picked before the data exists:

1. **Raise head count for the dial condition.** Under the forced design, S's
   positional resolution scales with head count alone. More heads = more rungs
   on the ladder. Frozen and identical across variants; changes the config, not
   the design.
2. **Re-scope H4d** to a task family less dominated by exact positional lookup,
   accepting that the difficulty dial then tests something narrower.
3. **Register the ladder cost as a finding** and grade H4a against D0p as
   planned while reporting the D0-vs-D0p gap as the price of the forced
   positional scheme — honest, and arguably a result worth having on its own.

Regardless of which cell wins, one line belongs in spec §2: **under the forced
positional design, S's positional resolution scales with head count alone**
(n_heads frequencies total, versus d_head/2 per head for standard RoPE). That
is an architectural constraint of the K3 kernel family and should be documented
whether or not it turns out to bind here.

---

## 7. Also unresolved, for the record

The δ=12 v1-LEAKY run (accuracy 0.099, floor 0.091) is **confounded, not
informative**. At δ=12 the non-binding floor sits almost exactly where "copy a
random present value" lands, so that number cannot distinguish suffix-shortcut
learning from no learning at all. My on-record prediction — that a model
learning the v1 leak converges to the suffix band ~0.92–0.97 — is **neither
confirmed nor refuted**, and stays open until a config that demonstrably solves
δ=0 is available to re-run it on. It should be re-tested at that point, since
it is cheap and it validates the shortcut-detection reasoning that produced the
v2 generator.

---

## 8. Recommendation

**No-go on GPU calibration.** Run cells 1 and 2 (and 3 only if both fail),
report the table in §4, and pick the §5 branch on evidence. Two and a quarter
hours of laptop background time, against the alternative of spending Colab
budget on the one axis that addresses neither leading hypothesis.
