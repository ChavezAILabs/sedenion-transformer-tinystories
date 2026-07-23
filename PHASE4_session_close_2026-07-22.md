# PHASE4_session_close_2026-07-22 — grid launch + 12/18 runs + mechanism read

**For:** Claude Code, next repo session
**From:** Claude (claude.ai chat), live with the owner through launch,
twelve runs, and the first diagnostic read
**Supersedes:** `PHASE4_colab_launch_log_2026-07-22.md` (written mid-X at
seed 1337; that file's §5 numbers are stale, its §2 incident log and §3
hotfixes remain accurate). Where the two disagree, this file wins.

---

## 1. State in one paragraph

The grid launched on Colab Pro A100 and ran **12 of 18 runs to
completion — seeds 1337 and 1338, all six variants each**. Seed 1339
stopped after D0p reached step 15260 when Colab compute units hit zero;
the owner is waiting for the free-tier reset rather than buying units
(deliberate, not blocked). **The completeness gate holds: nothing is
graded, the frozen §5 readout has not fired, and no verdict language
belongs anywhere in the repo yet.** Two hotfixes were applied live to
`phase4_grid.py`; the Drive copy at `MyDrive/p4_code/` is canonical
until synced back. Everything below the results line is descriptive,
two-seed, and explicitly ungraded.

## 2. What completed (Drive `MyDrive/p4_runs_ts/`)

Seeds 1337 and 1338, six variants each, all `diverged=False`:

| run | val 1337 | val 1338 | ppl512 1337 | ppl512 1338 | ppl1024 1337 | ppl1024 1338 |
|---|---|---|---|---|---|---|
| D1  | 1.6367 | 1.6428 | 8.67  | 14.46 | 24.63 | 35.06 |
| D0  | 1.6404 | 1.6463 | 7.38  | 7.82  | 18.68 | 19.46 |
| D0p | 1.6644 | 1.6670 | 12.55 | 6.89  | 35.05 | 24.41 |
| X   | 1.7638 | 1.7640 | 18.67 | 18.05 | 52.22 | 51.60 |
| S   | 1.7979 | 1.8014 | 12.14 | 11.22 | 39.81 | 34.38 |
| Q0  | 1.8241 | 1.8271 | 21.08 | 20.96 | 45.50 | 47.18 |

Wall-clock: dense family ~36 min/run (D1 ~40), S/X ~87 min/run.
Remaining seed-1339 runs ≈ 5.5 h on an A100 (D0p resumes mid-run from
`ckpt_last.pt`).

## 3. Observations — DESCRIPTIVE, TWO SEEDS, UNGRADED

**(a) In-distribution ordering is stable and unfavorable to S.**
D1 < D0 < D0p < X < S < Q0 on both seeds, tight across seeds (largest
seed-to-seed val drift 0.006). S trails D1 (H4a clause 1's comparator)
by ~0.16 both seeds — clause 1 is heading negative, and Q0 finishing
*below* both K3 variants kills the "attention barely matters on this
task" worry: the interaction is doing real work here.

**(b) The replicated observation: S beats X at ppl@512, twice.**
12.14 vs 18.67 (seed 1337) and 11.22 vs 18.05 (seed 1338) — margins of
6.5 and 6.8. X wins in-distribution by ~0.037 both seeds. So the
scrambled tensor fits better; the true tensor extrapolates better at the
first context doubling. This ordering is the single most robust
cross-seed pattern in the data.

**(c) Confounder that must appear in any writeup: the ladder costs
extrapolation.** D0 (standard 32-freq RoPE) vs D0p (frozen 6-freq
ladder), identical otherwise: ppl@512 7.38 vs 12.55 (1337) and 7.82 vs
6.89 (1338). Seed 1337 shows a large ladder penalty; seed 1338 shows
none (D0p actually better). S and X both carry the ladder, so the
defensible comparison for S is against **D0p**, never D0 — which is
exactly why the spec made D0 report-only. Do not let an S-vs-D0
comparison into any draft.

**(d) ~~ppl@1024 is high-variance across seeds; ppl@512 is not... Treat
the 512 rung as the evidential one and the 1024 rung as noisy.~~
CORRECTED 2026-07-23 (Claude Code, per `P4_AMENDMENT_A2_rung_posture.md`
A2.4 — flagged, not silently rewritten, per this project's standing
correction convention).** The original claim compared absolute
perplexity deltas, where 1024 naturally swings more because its values
run 2–3× larger (e.g. D0p's raw delta is 5.66 @512 vs 10.64 @1024).
Perplexity is ratio-scale, so cross-seed spread should be compared
*relatively* (max/min), and on that basis the ordering **reverses** on
the dense family (D0, D0p, D1 — the runs independent of the S/X
question): D0 1.06/1.04, D0p **1.82/1.44**, D1 **1.67/1.42** (@512/
@1024) — up to **82% at 512 vs 44% at 1024**, confirmed by independent
recomputation. **512 is the more variable rung, not 1024.** This matters
most because the original claim, if left standing, would have been
available to discount the H4a clause 2 negative (S loses to D0p at
ppl@1024 on both seeds) as "just noise" — exactly the place a discount
would be least defensible. Removing it makes that negative *cleaner*,
not weaker — the correct direction for a pre-registered study. D0p
35.05→24.41, D1 24.63→35.06 (absolute swings 30–40%) are still accurate
readings, just not evidence that the *rung* is noisy relative to 512.
H4a clause 2 reads ppl@1024 against D0p, where S loses on both seeds —
this grading is unaffected by the correction.

**(e) Mechanism, from `eval_log.jsonl` (the strongest finding of the
session).** Per-layer min(r²) and frac(r² < 1e-2), step 0 → final:

- **S seed 1337 L0:** min 0.1529 → **0.0013** (117×), frac<1e-2
  0 → 0.0033
- **S seed 1338 L0:** min 0.1382 → **0.0014** (99×), frac<1e-2
  0 → 0.0018
- **X seed 1337 L0:** min 0.0522 → 0.0502 (1.04×), frac<1e-2 → **0**
- **X seed 1338 L0:** min 0.0820 → 0.0375 (2.2×), frac<1e-2 → **0**

Every S layer drives min(r²) down 8–100×; no X layer moves it more than
~2×, and **X never places a single pair below r² = 1e-2 in either seed.**
Since the score is s = −γ·r², the manifold is the attend-here set: S
inhabits it, X does not. Layer structure: S's L0 shows five of six heads
with p5 dropping in both seeds and learns low γ (~1.2–1.7); deep layers
(L3–L5) show p5 *rising* with high γ (~3.1–3.7) in both variants. X only
ever exhibits the second regime — it has no manifold-seeking early
layer. **γ never flipped sign in any of 72 head-slots across four runs.**

**(f) Wall-clock exceedance (recorded, not graded).** S/X ~433 s/eval
interval vs dense ~181 s ⇒ ~2.4×, over the post-hoc 2× S/D0p bound the
grand summary will flag. Audited FLOP-invariant cost is 1.157×; the
residual is this einsum implementation on this hardware with TF32 off.
Owner's framing, endorsed: a systems question, not a property of the
mathematics.

## 4. Two structural findings about the pre-registration itself

Both belong in `RESULTS_phase4.md` as methodology notes, and both are
the freeze working as intended — it committed before anyone knew where
to look:

1. **The effect lives at a rung the rules don't grade.** H4a clause 2
   reads ppl@1024; the replicated ordering is at ppl@512.
2. **H4c as frozen may not discriminate.** The rule is a *direction*
   test with margin = 2× the across-seed std of step-0 p5 — and that std
   is ~0.004, because init statistics are extremely stable. Several X
   L0 heads decreased by more than that margin in both seeds. So H4c
   could return PASS for S *and* for X, while the genuinely
   discriminating evidence (min r², frac<1e-2 — 10–100× magnitude gaps)
   is logged as descriptive only. **Do not cite "H4c passed" as evidence
   of ZD-specificity.** Report the magnitude comparison and label it
   post-hoc.

## 5. Cheap, high-value work available before seed 1339 (no GPU)

1. **Shared-phase null preservation for the shuffled tensor.**
   `phase4_ladder_spotcheck.py` verified it for the true structure
   tensor. Run the same check on `shuffled_structure_tensor(seed)` for
   1337/1338/1339. If it breaks for X, that is a direct mechanistic
   account of the 512 result — relative-position nullity holding by
   algebra at any distance — rather than an inference from correlation.
   This is the single highest-value item on the list.
2. **Close the last prior-art thread** (`PRIOR_ART_REVIEW_zda.md` §7
   item 3: forward-citation traversal from Zoology and Reggiani). It is
   the only open condition on the novelty claim.
3. **Layer-0 ablation design** (costs one GPU run later, design now):
   zero or clamp S's manifold-seeking L0 heads at inference and re-run
   length-gen. If the 512 advantage dies with them, correlation moves
   toward cause; if it survives, the manifold-seeking is incidental.
   Worth pre-registering as an amendment *before* running it.

## 6. Action items (repo hygiene, unchanged from the launch log)

A1. Sync Drive `p4_code/phase4_grid.py` → repo; log both hotfixes dated.
    Fix 1: `torch.tensor([...], device=r2.device)`. Fix 2: pooled
    quantile replaced with sort-based nearest-rank (torch.quantile has a
    hard 2^24-element cap; the pooled r² tensor at grid dims is ~25.2M).
    Note the nearest-rank vs interpolation change in pooled p5/med —
    descriptive only; the graded per-head p5 path is untouched.
A2. Apply both fixes to `phase4_train.py` (carries bug 1 verbatim; would
    hit bug 2 at grid dims). Keep the two `diagnose()` blocks aligned.
A3. HANDOFF updates: Colab upload set is **four** files (include
    `sedenion_kernel.py`); the six-cell recovery stack; Drive
    eventual-consistency note (a reader VM saw `eval_log.jsonl` at 0
    bytes for many minutes while the writer flushed — mid-run cross-VM
    log reading is unreliable, and the "at most one eval interval lost"
    resume bound weakens to "one interval beyond whatever checkpoint
    bytes propagated"); wall-clock exceedance with the
    invariant-vs-implementation framing.
A4. Memory: update the phase-4 entries with launch state, the 12/18
    position, and §3(e)'s engagement asymmetry.
A5. Posterity: the owner saved a screenshot of S seed 1337's first
    evals — suggested `PHASE4_first_light_S_seed1337_2026-07-22.png`,
    with a provenance line if it enters the repo.

## 7. Standing orders

- **No grading, no verdict language, no `RESULTS_phase4.md` conclusions
  until 18/18 and the grand summary's own readout.** Descriptive
  reading of completed summaries is fine and encouraged.
- **No Phase 5 spec drafting yet.** The owner is deliberately treating
  Phase 5 as contingent on the readout, not on how the results feel.
  Write it cold, after the negative is written up, informed by the
  prior-art reading.
- Closed doors all still closed: no K1, no variant R, no β-style
  equivalence test for K3, no within-head multi-frequency positional,
  `ZD_PAIR` unchanged, γ/β never weight-decayed, entmax stays a
  non-graded instrument arm and needs P4M8/P4T9 before any run.

## 8. One-line state for CLAUDE.md

Grid 12/18 complete (seeds 1337–1338 full, 1339 pending Colab quota
reset); two GPU hotfixes to `phase4_grid.py` live on Drive pending repo
sync; in-distribution ordering D1<D0<D0p<X<S<Q0 stable across seeds with
S trailing D1 by ~0.16 (H4a clause 1 heading negative); S beats X at
ppl@512 on both seeds by 6.5–6.8 and drives min(r²) down 99–117× in L0
where X never moves it — replicated, descriptive, ungraded; completeness
gate holding.
