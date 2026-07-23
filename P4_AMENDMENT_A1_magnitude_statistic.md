# Phase 4 Amendment A1 — magnitude statistic for the engagement asymmetry

**Drafted:** 2026-07-23, with seeds 1337 and 1338 complete and seed 1339
**not yet run** (status as of drafting — not independently confirmed
current at freeze time; see `P4_AMENDMENTS_HANDOFF.md` verification log
in `HANDOFF.md`).
**Status: FROZEN 2026-07-23 by Claude Code**, pending confirmation from
the owner that seed 1339 had not yet started at freeze time (see
freeze-time caveat in `HANDOFF.md`). A1.1 slot filled; A1.4 arithmetic
verified correct against `PHASE4_session_close_2026-07-22.md` §3(e) —
**not** independently checked against raw `eval_log.jsonl`, which is not
present in the local repo (lives on Colab Drive per session-close §6
item A1's sync note). This is one hop of provenance short of the
amendment's own ideal ("recompute from the logs") and is stated here
rather than silently treated as equivalent.
**Author:** Claude (claude.ai chat); verified/filled/frozen by Claude
Code.

---

## A1.0 What this amendment is, and what it is not

`PHASE4_session_close_2026-07-22.md` §4.2 records that H4c as frozen may
not discriminate: it is a *direction* test with a margin of ~0.004, and
several X L0 heads cleared that margin in both seeds. H4c could
therefore return PASS for S **and** for X, while the genuinely
separating evidence — the 10–100× gaps in min(r²) and the fact that X
places no pair below 1e-2 — sits in the log as descriptive only.

This amendment does **not** fix that by adding a hypothesis. Adding a
fourth graded hypothesis after seeing two seeds would be exactly the
reinterpretation the freeze exists to prevent.

What it does instead: fix the **exact form** of the descriptive
magnitude report — statistic, comparison, table shape, and the bar for
calling the pattern held — while still blind to a third of the data.

The result remains post-hoc relative to the pre-registration. But
"specified blind to seed 1339" is a materially stronger provenance than
"specified after seeing everything," and it is available for free right
now and never again. That is the entire purpose of this document.

**Standing constraint, unchanged:** nothing here is graded, nothing here
counts toward H4a/H4b/H4c, and no verdict language follows from it.

## A1.1 Definition of r² — FILLED 2026-07-23 by Claude Code

**Correction to the slot instruction:** the definition does not live in
`phase4_grid.py` Section 1 — unlike `zda_grid.py` (Phase 1–3), which
bundles a verbatim copy of the validated layer code, `phase4_grid.py`
**imports** `phase4_layers`/`phase4_model` rather than bundling
(documented deviation, `CLAUDE.md` Architecture section). The canonical
definition is in `phase4_layers.py`'s module docstring and
`K3Attention.scores()`.

Verbatim from `phase4_layers.py` (module docstring, lines 14–17):

```
r2_ij = |q_i (x) k_j|^2 / (|q_i|^2 |k_j|^2 + eps_div)
score = -gamma_h * r2_ij       (mandatory: no parameter setting
                                recovers dot-product attention)
```

**The product is the ladder-rotated form, not raw q_i ⊛ k_j** —
confirmed by reading `K3Attention.scores()` directly (lines 104–126):
`q, k = self._rotate(q, pos), self._rotate(k, pos)` happens *before*
the sedenion product `prod = einsum("bhti,bhsim->bhtsm", q, k_rot)`,
where `q`/`k` at that point are already `R_8(ω_h·pos)`-rotated. So
`r2_ij` is computed on `R_8(ω_h·pos_i)·q_i` and `R_8(ω_h·pos_j)·k_j`,
per head, per the frozen ω_h ladder — not on the unrotated projections.
`eps_div = 1e-12` (guards zero-norm only, confirmed below every
diagnostic bin per the docstring). Score relation `s = −γ_h·r²_ij` is
exact — confirmed by direct code read of `scores()` line 121
(`s = -self.gamma.view(1, -1, 1, 1) * r2`), not just assumed from
session-close §3(e).

Slot filled. Everything below is unblocked.

## A1.2 The two statistics

### D — decades of descent (primary)

For variant v ∈ {S, X}, seed s, layer ℓ:

```
D(v, s, ℓ) = log10( min r²(step 0) / min r²(final) )
```

- `min r²` is the minimum over all pairs in the layer's eval batch, as
  already logged per eval interval in `eval_log.jsonl`.
- **`final` means the last eval interval present in `eval_log.jsonl`
  for that run** — not "end of training." Stated explicitly because
  D0p resumes from checkpoint and §6/A3 weakens the resume bound.
- Reported in decades, two decimal places.

**Why a ratio and not the final value.** The step-0 baselines differ
substantially between variants: S L0 starts at 0.1529/0.1382, X L0 at
0.0522/0.0820. X begins roughly 2–3× closer to zero. Comparing final
absolute minima would flatter X for a property of its initialization.
The ratio is scale-free and asks the question actually at issue — does
training *drive* the quantity down.

### F(τ) — sub-threshold mass curve (secondary)

```
F(v, s, ℓ, τ) = fraction of pairs with r² < τ, at final
```

reported as a **curve over τ ∈ {1e-1, 1e-2, 1e-3, 1e-4}**, never at a
single τ.

**Why a curve.** The 1e-2 threshold in session-close §3(e) was chosen
after looking at the data. Freezing a post-hoc threshold would preserve
the degree of freedom rather than remove it. Reporting the whole curve
eliminates the choice: no threshold can be selected to favour a
conclusion, because all four are always shown.

### A1.2.1 Why these two and not the p5 statistics

Both D and F are **order statistics** — a minimum and a count. Neither
is affected by hotfix 2 (A1 in the session-close: `torch.quantile`
replaced with sort-based nearest-rank because the pooled r² tensor at
grid dims exceeds the 2^24 element cap).

Pooled p5 and med *did* change method mid-flight. Any magnitude
statistic built on pooled quantiles is therefore not cleanly comparable
across the full grid. D and F are the strongest magnitude statistics
available that are provably immune to that change. This is a
correctness argument, not a preference, and it should be stated in the
write-up.

## A1.3 Reporting format — fixed now

1. **Full table, no selection.** 6 layers × 2 variants × 3 seeds for D;
   the same shape × 4 thresholds for F. L0 is the headline but reporting
   only L0 would be selection after the fact. Deep layers (L3–L5) show
   the opposite regime in *both* variants and must appear.
2. **Per-seed values always.** Never a mean across seeds, never a
   standard error, never a p-value or confidence interval. n = 3.
   Session-close §3(d) already showed ppl@1024 swinging 30–40% on
   identical configs; the same caution applies here.
3. **Across-seed spread as [min, max]**, not ± anything.
4. **Primary comparison is within-seed, same-layer, S vs X**, reported
   as the separation `D_S − D_X` in decades. This is the control
   comparison and the only one carrying attribution weight.
5. **Replication stated as "k of 3."** If seed 1339 dissents, it is
   reported as 2 of 3 with the dissenting value shown. Never aggregated
   in a way that conceals a dissenting seed.

## A1.4 Bar for "the pattern held" — set blind to seed 1339

These are descriptive bars, not decision rules. They exist so that the
sentence written after 1339 was chosen before 1339.

At **layer 0**, per seed:

| # | pattern | bar |
|---|---|---|
| P1 | S descends | D(S, ·, L0) ≥ 1.5 decades |
| P2 | X does not | D(X, ·, L0) ≤ 0.5 decades |
| P3 | separation | D(S,·,L0) − D(X,·,L0) ≥ 1.0 decade |
| P4 | X reaches no nulls | F(X, ·, L0, 1e-2) = 0 |

**Honest provenance of these numbers.** They were set by looking at
seeds 1337 and 1338, which are already seen and cannot be unseen. They
are blind only to seed 1339. Values implied by the two completed seeds:

| | 1337 | 1338 |
|---|---|---|
| D(S, L0) | log10(0.1529/0.0013) = **2.07** | log10(0.1382/0.0014) = **1.99** |
| D(X, L0) | log10(0.0522/0.0502) = **0.02** | log10(0.0820/0.0375) = **0.34** |
| separation | **2.05** | **1.65** |

Both seeds clear all four bars with margin. The bars are set below the
observed values but not trivially so — X seed 1338 at 0.34 is within
0.16 decades of the P2 bar, which is deliberate: if seed 1339's X moves
more than the others, P2 should be capable of failing.

**Scope note.** P1 is scoped to L0 only. Session-close §3(e) reports
every S layer descending 8–100× (0.9–2.0 decades), so a 1.5-decade bar
applied to all layers would fail on the weakest S layer by
construction. All layers are reported; only L0 carries a bar.

## A1.5 γ sign census

Session-close §3(e): γ never flipped sign across 72 head-slots in four
runs. Pre-specified extension:

- With seed 1339 the census becomes **108 head-slots across six runs**.
- Reported as a fraction, e.g. "0 of 108."
- **Any sign flip is reported prominently and individually** — variant,
  seed, layer, head — not absorbed into a summary count or dismissed as
  noise. A single flip is informative given 72 consecutive non-flips.

## A1.6 What this statistic cannot support

Written now, to be quoted verbatim in the write-up:

- It is **post-hoc** relative to the pre-registration and is labelled as
  such wherever it appears.
- It is **descriptive and correlational**. It shows that S occupies the
  low-r² manifold and X does not. It does **not** show the 512-rung
  advantage is caused by that occupancy.
- It **does not substitute for H4c** and must never be cited as H4c
  passing. Per session-close §4.2, "H4c passed" must not be offered as
  evidence of ZD-specificity in any case.
- The only route from this correlation toward cause is the **layer-0
  ablation** (session-close §5 item 3), which must itself be
  pre-registered as an amendment before it is run.
- It is **not** grounds to revisit H4a's rung choice. That is a separate
  pre-commitment (Amendment A2, not yet drafted).

## A1.7 Freeze procedure

1. ~~Claude Code fills the A1.1 slot from the repo.~~ **DONE** — see
   A1.1, sourced from `phase4_layers.py` directly (not `phase4_grid.py`,
   corrected).
2. ~~Verify the four bars against `eval_log.jsonl`~~ **PARTIAL.** The
   arithmetic in A1.4 (the four `log10` computations and the two
   separation values) was recomputed independently by Claude Code and
   matches exactly. The *source* r² values (0.1529, 0.0013, 0.0522,
   0.0502, 0.1382, 0.0014, 0.0820, 0.0375) were checked against
   `PHASE4_session_close_2026-07-22.md` §3(e), where they match exactly
   — but that document is itself one hop from `eval_log.jsonl`, which is
   not in the local repo (Colab Drive only, per session-close §6 A1).
   **Full verification against the raw log is still owed** and should
   happen whenever the repo next has Drive access (the same sync session
   session-close §6 item A1 already calls for).
3. **DONE** — file dated and frozen 2026-07-23. (No git repo in this
   project — see `HANDOFF.md`; the dated freeze header plus the
   `HANDOFF.md` log entry is the audit trail in place of a commit hash.)
4. Recorded in `HANDOFF.md` in place of a commit hash (no git repo).

**Ordering caveat, stated plainly:** this file was frozen before Claude
Code had independent confirmation of seed 1339's run status — the
owner's message that delivered these amendments predates this
verification pass, and the assistant cannot observe Colab state
directly. If seed 1339 had already started or completed by the time
this freeze happened, the "blind" property is void and this amendment
must be re-labelled ordinary post-hoc analysis, per its own rule. Flagged
to the owner for confirmation; not resolved unilaterally.

---

## Next in this series (not yet drafted)

- **A2 — rung posture.** The replicated effect is at ppl@512; H4a clause
  2 grades ppl@1024. Fix how both are reported, blind to 1339.
  Deciding after 1339 looks like rung-shopping even when it is not.
- **A3 — layer-0 ablation design.** Zero or clamp S's manifold-seeking
  L0 heads at inference, re-run length-gen. Budget ~20 compute units.
  Must be frozen before running.
