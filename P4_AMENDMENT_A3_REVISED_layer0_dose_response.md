# Phase 4 Amendment A3-revised — layer-0 dose–response design

**Supersedes:** `P4_AMENDMENT_A3_layer0_ablation.md`. The original file is
**kept in place, unmodified** — this is an addition, not an edit, per
`P4_AMENDMENTS_HANDOFF.md` I3's own instruction ("supersede rather than
edit").
**Provenance, stated exactly as required:** (a) A3 was frozen 2026-07-23
but **never executed** — no arm beyond harness design existed, no data
was generated or seen; (b) the γ=0 selection argument in the original
A3.1 (chosen "over" head-zeroing because it supposedly "leaves capacity
and content-based attention intact") was found **void** before
execution — γ=0 yields uniform, content-blind attention, confirmed both
by code read and by an empirical script run on a live `K3Attention`
instance (see I3a below), so it is not obviously less capacity-
destroying than zeroing, and the stated ground for preferring it does
not hold; (c) no ablation data of any kind informed this revision — the
only numbers referenced below are the two-seed grid summary values
already on record in `PHASE4_session_close_2026-07-22.md` §3(e), same
as the original A3.
**Status: FROZEN 2026-07-23 by Claude Code**, design only — harness not
yet built (same division as the original A3: build/validate now, on
1337/1338 checkpoints; execute and report only at 18/18).
**Author:** design proposed by Claude (claude.ai chat) in
`P4_AMENDMENTS_HANDOFF.md` I3b; evaluated for repo fit, one ambiguity
resolved, and frozen by Claude Code.

---

## R.0 Why A3 was reopened (not a violation of "don't revise after data")

A3 produced no output of any kind. Revising an untested design is
ordinary practice; revising a design after seeing what it returns is the
violation this project's standing orders exist to prevent. The
distinction is load-bearing and is why this is `A3-revised`, dated and
provenanced, rather than a silent edit to the original file.

## R.1 I3a — the four facts, verified before any redesign

Per the routing doc's own order ("report the four facts before
designing"), verified by Claude Code before R.2 was written:

1. **γ is per-head and multiplicatively scalable at inference without
   touching any other parameter — TRUE.** Confirmed by direct read of
   `phase4_layers.py`: `self.gamma = nn.Parameter(torch.full((n_heads,),
   gamma_init))`, used only in `scores()` as
   `s = -self.gamma.view(1,-1,1,1) * r2`. Scaling `gamma[h] *= m` for a
   subset of head indices touches nothing else — not `wq`/`wk`/`wv`/
   `wo`, not other heads' γ, not other layers.
2. **γ's init value is 1.0**, per head, for every variant. Confirmed by
   reading `K3Attention.__init__`'s default (`gamma_init: float = 1.0`)
   and `phase4_model.py`'s constructors for S and X (`K3Attention(d_model,
   n_heads)` and `K3Attention(d_model, n_heads, tensor=...)`) — neither
   passes `gamma_init`, so both use the class default. This means
   **m=1.0 in the sweep below is not merely "close to the trained
   value," it is the literal init point** for every arm before training
   diverged the heads apart — worth stating in the eventual writeup.
3. **Confirmed empirically, on a live model, not by reading code**: with
   `gamma[h] = 0`, that head's attention becomes exactly uniform over
   the causal window and exactly content-blind (identical attention
   regardless of input content), while every other head is bit-for-bit
   unaffected. Script: `gamma_zero_check.py` (run this session; not yet
   copied into the repo as a permanent test since it's a narrow sanity
   check subsumed by A3.2 once the real harness exists — the four
   checks below cover the same ground and will be the permanent,
   rerunnable version). All four sub-checks passed with **exact** zero
   deviation (not "small" — literally `0.000e+00` in every case,
   consistent with `s = -0 * r2 = 0` having no floating-point
   approximation to begin with).
4. **From `eval_log.jsonl`: NOT independently re-derived** — raw logs
   are Colab-Drive-only and not present in this repo (same limitation
   stated throughout `HANDOFF.md` §4a for A1/A2). The values used below
   (L0 ≈ 1.2–1.7, L3–L5 ≈ 3.1–3.7) are as recorded in
   `PHASE4_session_close_2026-07-22.md` §3(e), already independently
   cross-checked once this session (A1/A2 work) and not re-derivable
   further without Drive access.

All four facts hold as assumed. Proceeding to the design.

## R.2 The intervention

Two arms, both inference-only, both multiplicative on the **learned**
per-head γ at inference (no retraining):

### R.2.1 Sweep arm (primary)

For m ∈ {0, 0.25, 0.5, 1.0, 2.0}: set L0's γ to `m * gamma_learned`,
where `gamma_learned` is the checkpoint's actual trained per-head γ
values at L0 (six values, one per head). Applied to **both S and X**,
evaluated at **both rungs** (ppl@512, ppl@1024), on **all three seeds**.

- `m = 1.0` is the intact model — arm I is literally `m=1.0`, not a
  separate code path. The harness validates itself at a known point on
  its own curve for free.
- `m = 0` is retained (it is the original A3's arm P for S and arm C1
  for X) but is now one endpoint of five, not the whole experiment.
- This folds the original A3's separate **C1 (X, identical
  intervention)** into the design directly: X gets its own m-curve,
  computed the same way, not a bolt-on control. C1 is retained as a
  concept — "does X degrade comparably to S under the same
  intervention" — but is now read off the two curves at matching m,
  not run as a separate arm.

### R.2.2 Clamp arm (targets the actual finding, not "is there a gate")

Set L0's γ to the deep-layer learned value, **per seed and variant**.

**Ambiguity in the original proposal, resolved here (Claude Code):**
"median over L3–L5" could mean per-head (matching head index h in L0 to
the same index h in L3–L5) or pooled (one scalar from all 18 values,
applied uniformly). **Fixed choice: pooled.** There is no evidence in
this repo that head index identity is consistent across layers —
heads are independently learned per layer, and nothing establishes that
"head 2" plays the same role at L0 and L3. Per-head matching would
smuggle in an unexamined assumption; pooling (median of all 3 layers × 6
heads = 18 values, one scalar, applied to all six L0 heads uniformly)
makes no such assumption and is the more defensible default. Stated here
so it is fixed **before** execution, not chosen after seeing which
choice looks better.

This targets the phenomenon actually observed in session-close §3(e) — L0's
*low-γ* (~1.2–1.7) manifold-seeking regime vs. the deep layers' *high-γ*
(~3.1–3.7) regime — rather than testing "does having a gate at all
matter," which is what m=0 tests.

### R.2.3 Retained from the original A3, unchanged

- **C2 (layer-identity control):** γ=0 on 6 heads sampled from L3–L5,
  fixed rule (lowest head index first, two per layer). Still needed —
  the sweep/clamp arms test *which value* of γ at L0 matters, C2 tests
  whether disrupting *any* layer's γ (not specifically L0's) produces a
  comparable effect.
- **Arm I** (m=1.0, both S and X, all seeds) must reproduce recorded
  grid values (val, ppl@512, ppl@1024) to floating-point tolerance —
  gating precondition, unchanged.
- **A3.2's four harness checks**, unchanged in substance from the
  frozen (corrected) A3 — check 1's wording already states the correct
  target ("gate path removed" = `s := 0` exactly, i.e. uniform
  attention, not "standard attention"). Extend check 1 to also verify
  the **general** multiplicative case: `gamma[h] = m * gamma_learned[h]`
  produces `s = -m * gamma_learned[h] * r2` exactly, for arbitrary m,
  not just m=0.

## R.3 Metrics — unchanged in form from the frozen A3

Per arm point (m or clamp), per seed: val loss, ppl@512, ppl@1024.
Primary quantity, the **S–X gap**, now a function of m:

```
G(m, r, s)  = ppl_X(m, r, s) - ppl_S(m, r, s)     [r = rung, s = seed]
```

with `G(1.0, r, s)` reproducing the intact-model gap already on record
(6.53/6.83 @512, 12.41/17.22 @1024, seeds 1337/1338).

**Gap retention, generalized:** `R(m, r, s) = G(m, r, s) / G(1.0, r, s)`.

## R.4 Pre-specified reading — fixed now, blind to every ablation number

**Curve shape, the primary read:**

| pattern | reading |
|---|---|
| R(m) flat across m ∈ {0, 0.25, 0.5, 2.0} (all within [0.75, 1.25] of 1.0) | L0 γ-magnitude incidental to the S–X gap |
| R(m) monotonically increasing toward 1.0 as m→1 (i.e., declining as m→0) | consistent with a causal role, subject to R.5 below |
| non-monotone / noisy | report the curve numerically, no label |

**Endpoint labels, generalized from the frozen A3's binary rule and
updated per the routing doc's explicit instruction** ("R < 0.25 is now
very weak"):

- `R(0) > 0.75`: gap survives complete L0 gate removal — a strong
  negative on L0-manifold-seeking specifically (not confounded by
  adaptation dependence — see R.5).
- `R(0) < 0.25`: **very weak, more so than in the original design.**
  m=0 is now known to be a heavy, content-blind intervention (R.1 item
  3), not a mechanism-only removal — a collapsed gap at m=0 alone is
  barely more informative than it was before, and must lean on the
  intermediate points to mean anything.
- **The intermediate points (m ∈ {0.25, 0.5, 2.0}) carry the reading**
  whenever they are available and val hasn't degraded sharply there —
  this is the actual payoff of the redesign over a single ablation
  point.

**Clamp-arm reading:** report `R_clamp(r, s)` the same way. A clamp
result that tracks the sweep's m≈0.3–0.5 region (interpolating where the
clamp value falls relative to `m=1.0 * gamma_learned`) is a consistency
check between the two arms, not an independent third finding — state it
as such, don't double-count.

**Additional pre-specified checks, generalized to per-arm granularity
(this directly addresses the routing doc's I3 concern that the original
whole-design abort condition was "likely to fire on every arm"):**

- **Per-point val check:** if in-distribution val degrades sharply
  *at a specific m*, **that point** is reported as uninformative and
  excluded from the curve reading — it does not void the other points.
  If val degrades sharply at *every* m including m=1.0 (which should be
  impossible, since m=1.0 is arm I and must reproduce the intact grid
  value — if it doesn't, the harness itself is broken, not the model),
  stop and treat it as a harness bug per A3.2 check 3.
  **Registered expectation, stated in advance:** m=0 is the most likely
  point to trip this check, precisely because it is now known to be
  content-blind and maximally perturbative — a val cliff exactly at
  m=0 and nowhere else is itself informative (it locates how much of
  the intact model's performance depends on L0 doing *some* selective
  routing at all, distinct from the S-vs-X question), not a design
  failure.
- **C2 check, whole-design scope, unchanged:** if C2 (deep-layer m=0)
  degrades comparably to the L0 sweep's m=0 point, layer identity is not
  doing the work and no L0-specific claim is supported. This one still
  voids the L0-specific reading if it fires, since it's testing the
  premise the whole design rests on.
- **Generic-gate-removal check, now read directly off the two curves:**
  if X's curve shape closely tracks S's curve shape (not just its m=0
  point), the effect is generic-intervention response, not
  manifold-seeking-specific — read this from the *shapes*, not a single
  point comparison, which is the sweep's main advantage over the
  original binary C1.

## R.5 The confound, and the asymmetry — carried forward, one clause updated

Unchanged from the original A3.6: this is an inference-time intervention
on a model trained with the gate present, so degradation conflates "the
mechanism does work at inference" with "the rest of the model depends on
the gate being present" (adaptation dependence). The clean version
(retrain S without L0 gating) remains out of budget and is named as the
experiment this substitutes for.

**Updated per R.4:** `R > 0.75` (gap survives) remains a reasonable
negative, not confounded by adaptation dependence the same way a
collapse would be. `R < 0.25` was already labeled "weak, suggestive
only" in the original A3; per the routing doc, it is now **very weak**
specifically at m=0, since a heavy content-blind intervention killing
the gap tells us little beyond "the model needs *some* L0 selectivity
to function at all" — the intermediate m points are what would let a
declining-R reading mean more than that.

## R.6 Cost — still inference-only, more points, still cheap

Still zero training, zero gradients — evaluation passes over existing
checkpoints only. More points than the original binary design (5 sweep
values + 1 clamp value, ×2 variants, ×2 rungs, ×3 seeds, plus C2 and arm
I), but cost scales with eval batches, not model size or steps — still
plausibly free-tier viable per the original A3.7 estimate (order
5–15 units), to be confirmed the same way: one arm-I reproduction on a
free-tier T4 before assuming either way.

## R.7 What may not be claimed — unchanged from the original A3.8

Descriptive and post-hoc; not H4a/H4b/H4c and does not bear on them;
addresses S-vs-X only, says nothing about S vs D0p or S vs D1; no
outcome establishes ZD-specificity alone (still requires the controls
read together, C2 in particular now that C1 is folded into the sweep);
a favourable result does not license a Phase 5 spec.

## R.8 Freeze procedure

1. **DONE** — I3a's four facts verified (R.1), one empirically (script
   run, not just code read).
2. **DONE** — design evaluated for repo fit, adopted with one resolved
   ambiguity (clamp-arm pooling, R.2.2).
3. **Not yet done** — harness build: extend A3.2's checks for the
   general multiplicative case; implement the sweep/clamp/C2/arm-I
   loop. Buildable now on 1337/1338 checkpoints, no ablation number
   generated or inspected in the process.
4. **DONE** — file dated and frozen 2026-07-23 (design only). No git
   repo existed before this session; one now does
   (`P4_AMENDMENTS_HANDOFF.md` I2) — this file's freeze should get its
   own commit, separate from the initial commit that already contains
   the original (unrevised) A3, per I2's "every future freeze gets its
   own commit" instruction.
5. Recorded in `HANDOFF.md` §4a alongside the existing trail.

**Ordering caveat — RESOLVED 2026-07-24, LABEL: POST-HOC, NOT BLIND**
(same resolution and correction as A1/A2/A3 — see `HANDOFF.md` §4d and
A1's caveat for the full reasoning). The owner confirmed the seed-1339
log seen 2026-07-24 is the same 2026-07-22 historical run (D0p stalled
at step 15260/18311) — **that date is before this file's own draft/
freeze date (2026-07-23/24), same as A1/A2/A3**, so the same
weaker-label rule applies: partial seed-1339 data existed before
drafting, whether or not it was observed. **Label: post-hoc, not
blind**, for the whole series uniformly.
