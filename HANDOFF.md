# HANDOFF.md — ZDA Experiment: session-to-session pickup

**For:** whoever picks this up next
**From:** Claude Code (Sonnet 5), session ending 2026-07-21; §3/§4
merged 2026-07-22 by Claude Code (Fable 5) from the 2026-07-22
chat-session addendum (Claude, claude.ai)
**Owner:** Paul Chavez / Chavez AI Labs

**Start with `CLAUDE.md`** — it's the living contract (status paragraph,
invariants, commands, "closed doors") and gets loaded automatically every
session. This file is the narrative companion: what actually happened,
in what order, and what to do next. If the two ever disagree on a fact,
trust `CLAUDE.md` — it's been kept current, this file is what falls
behind between refreshes.

---

## 1. Where things stand

**Phase 1–3: done, resolved, all negative.** `ZDA_experiment_spec.md` /
`RESULTS_phase1.md` / `RESULTS_phase2_smoke.md` / `RESULTS_phase3.md`.
The additive zero-divisor gate on standard attention never got recruited
(β traces show the control frame recruited *harder* than the true ZD
frame). One exploratory keeper survived: 16-way PHM weight sharing gives
~3× better length-generalization perplexity than dense, at modest
in-distribution cost — algebra-independent (a shuffled-tensor control
shows the same effect), so it's a weight-sharing finding, not a
zero-divisor one.

**Phase 4: in progress**, motivated directly by Phase 3's null — the old
gate never mattered because the *score itself* was ZD-blind. K3 (norm-
ratio kernel) fixes that by construction: its interaction term is exactly
the deviation from norm-multiplicativity, not an add-on. Read
`ZDA_phase4_spec.md` (**v1.0, FROZEN 2026-07-22** — from here, changes
only by logged, dated amendment) and `PHASE4_kernel_memo.md` before
touching any `phase4_*.py` file.

**Gate 5 (dial task) is closed, negatively, as of 2026-07-21:**
`PHASE4_gate5_outcome.md` has the full record — an LR sweep, an extended
step budget, and a from-scratch MQAR-standard multi-query redesign all
failed to reach the pre-registered 0.89 bar at δ=0. H4d is recorded as
**pre-registered but not executed**, not dropped, not silently replaced.
**Phase 4 now proceeds on the anchor task only (H4a, H4b′, H4c) — this
is unblocked and ready.** Resumption condition for H4d, pre-registered:
an H4a surprise positive makes the dial worth resuming.

## 2. What happened 2026-07-21 (Claude Code session, all at the owner's direction)

Four threads, each self-contained; read the referenced doc for the full
account rather than re-deriving from this summary:

1. **Read Reggiani (arXiv:2411.18881)** — the sedenion zero-divisor
   variety is isometric to G₂. Repo-verified against the paper's own
   LaTeX source, not a PDF summary (one had already gotten a dimension
   wrong). `ZD_PAIR` turned out to be literally row 52 of the paper's
   own 84-pair table. Built a new, sharper X-inequivalence certificate
   on top of the theorem (`phase4_reggiani_certificate.py`) — the true
   tensor's null pairs sit at *exactly* σ_min=√2 with zero singularities
   (G₂ acting transitively); the shuffled-tensor draws show scattered
   values and genuine singularities at 3–9% of their null pairs. Full
   account: `PHASE4_reggiani_reading.md`. **Proposed but not applied:**
   a one-sentence addendum to spec §9 item 1's interpretation
   constraint — held for owner sign-off. *(Update 2026-07-22: approved
   and applied in spec v1.0 — see §3.)*
2. **Two of three literature follow-ups closed** (`PRIOR_ART_REVIEW_zda.md`
   §8): no non-associative/zero-divisor attention work exists anywhere
   (five search angles). One real find: GATr (the closest prior art) is
   itself built on a degenerate Clifford algebra and its own paper admits
   its attention score "ignores the 8 dimensions involving e₀" — patched
   with a dualization trick, never exploited as signal. Separately, no
   paper connects within-projection Kronecker/PHM weight sharing to
   length generalization — closes the open condition on claiming the
   Phase 3 V1/V4 finding as genuinely novel. **Still open:** forward-
   citation traversal from Zoology and Reggiani (§7's third gap).
3. **E₈/G₂ Lie-algebra connections verified** (`LIE_ALGEBRA_CONNECTIONS.md`).
   The owner dropped in `e8_weyl_orbit_unification.lean` from their own
   separate, already-published project (`canonical_six_publication/`,
   Zenodo DOI 10.5281/zenodo.18793480 — see
   `canonical-six-publication-reference` in memory). Independently
   reproduced its three theorems in exact rational arithmetic
   (`verify_e8_weyl_orbit.py`) — `ZD_PAIR` sits in a single E₈ Weyl
   orbit, a three-way convergence with Reggiani's table and this
   project's own pre-Phase-3 adoption. **One thing flagged for the
   owner's other project, not acted on here:** their `g2_family_24_
   investigation.lean` has an open "G₂ invariance" stub blocked on
   Mathlib — Reggiani's paper looks like the missing citation. Owner's
   to pick up later, in that repo, not this one.
4. **GPU/TF32 hygiene confirmed on real hardware** (spec §9 item 2, now
   DONE). Found and fixed a real gap first: `phase4_train.py` had no
   TF32 guard at all (PyTorch defaults it on for CUDA; `zda_grid.py`,
   Phase 1–3, explicitly *enables* it — wrong default to inherit for
   K3's score path). Built `phase4_gpu_tf32_confirm.py` (+ a paste-ready
   Colab-cell twin), caught and fixed two of my own bugs via CPU dry-run
   and a first failed Colab attempt before trusting real GPU output, then
   ran it clean on the owner's Colab Pro (Tesla T4): fp32/TF32-off is
   faithful to ε=1e-5 with a live gradient — **guard confirmed working.**
   TF32-on vs off were bit-identical at this shape (inconclusive on
   whether this contraction ever hits tensor cores at Phase 4 sizes;
   guard kept regardless, it's free). fp16 floors at ε=1e-4. **bf16
   showed no floor at all down to ε=1e-5**, on two independent runs —
   corrects an earlier informal "~1e-2, unusable" estimate that was
   never saved as a rerunnable script. All recorded in spec §9 item 2
   with the discrepancy flagged, not silently overwritten.

## 3. What happened 2026-07-22 (chat session, claude.ai, outside the repo)

This session ran in chat — three docs plus `zda_grid.py` and
`phase4_train.py` uploaded, no repo access. It landed in the repo via
three delivered files: `ZDA_phase4_spec.md` (v1.0), `phase4_grid.py`,
and the handoff addendum merged into this section. All owner-directed,
all owner-approved. (`spec_v1_freeze_diffs.md`, cited in the spec's
changelog, was the chat-side diff document the owner approved — it lives
in that chat session, not in this repo.)

1. **The three v1.0-freeze TBDs resolved and signed off:** γ frozen as
   s = −γ·r², per-head, unconstrained, init 1.0; ω_h ladder frozen as
   geometric with task-pinned endpoints (ω_0 = 1 → ω_{H−1} = 2π/1024,
   H = 6, worked table in spec §2); H4c frozen on per-head p5(r²)
   computed on causal support, pass = ≥1 (layer, head) crossing its
   step-0 value by 2× pooled step-0 std in all 3 seeds. §6's metric
   frozen to match. The two held 2026-07-21 addenda (Reggiani
   smoothness certificate → §9.1; GATr degenerate-metric sentence → §2)
   were approved and applied.
2. **Entmax scoped, owner's hunch, option 2 of 3 discussed:**
   entmax-1.5 as instrument arm 2 (spec §3) — α fixed, S/D0p only,
   1 seed, anchor task, post-grid, vendored + mirror-first (P4M8/P4T9
   required before any run), support-size diagnostic added, explicitly
   barred from all H4 verdicts. Option 1 (grade it — doubles the grid)
   and option 3 (defer, recorded) were considered and set aside.
3. **Spec frozen and bumped to v1.0** (`ZDA_phase4_spec.md`): all seven
   diff blocks from `spec_v1_freeze_diffs.md` applied by surgical edit
   to the uploaded v0.3 — untouched text preserved verbatim; H4d's own
   TBDs deliberately survive (retained-verbatim rule). Gate 6 marked
   DONE 2026-07-22.
4. **`phase4_grid.py` built** — the anchor-grid orchestrator that did
   not exist. Follows the `zda_grid.py` pattern (match table →
   preflight → sequential variant×seed with atomic resume → grand
   summary ending in the frozen §5 readout). Three documented
   deviations: imports `phase4_layers`/`phase4_model` instead of
   bundling (verbatim-copy rule can't be honored by retyping absent
   files); TF32 hard-disabled on CUDA, no escape flag (spec §9.2); no
   β=0 preflight — replaced with §7 anchors (ZD annihilation, γ
   init/no-decay, K1-guard-alive, causality, ladder verification,
   T=1024 forward). Reuses the Phase 3 `data_ts` cache. Its header
   carries the in-repo integration checklist (`DIMS_TUNED` /
   `FLOPS_AUDITED` sentinels, ladder wiring, T=1024 forward).

**Test coverage, honestly stated:** no torch existed in the chat
environment. The new decision logic (frozen H4c all-seeds-same-head
rule incl. the 2-of-3-seeds negative case, H4a two-clause + conditional
H4b′ arithmetic, resume-safe last-line-per-step log dedupe, ladder
endpoints) was EXECUTED against synthetic run outputs and passed. The
trainer/preflight/data paths are syntax-checked only — step 5 below is
their first real execution.

## 4. Ordered path to launch (each gates the next)

1. ~~Land the three files. Merge the addendum; refresh `CLAUDE.md`.~~
   **Done 2026-07-22** (this merge).
2. ~~Residual factual verifications.~~ **Done 2026-07-22, both PASS**
   (recorded in the spec changelog): (a) `phase4_reggiani_certificate.py`
   rerun live reproduced the §9.1 figures exactly (σ_min = √2 at
   336/336, 0 singularities; X seeds 3.1%/7.6%/9.1%); (b) rel-prop +
   shared-phase null preservation hold at all six frozen ω_h to ~7e-15
   (`phase4_ladder_spotcheck.py`, new, rerunnable). No v1.0.1 needed.
3. ~~Wire the frozen ladder into `phase4_layers.py`.~~ **Done
   2026-07-22:** new `frozen_omega_ladder()` in `phase4_layers.py` is
   the single source of truth (same formula as
   `phase4_grid.omega_ladder`); K3Attention's per-head frequency buffer
   renamed `freqs` → `omega_h` and defaulted to the frozen ladder
   (NOTE: the state-dict key changed, so pre-freeze smoke checkpoints
   no longer load into the new code — archived run outputs themselves
   are unaffected); ladder-mode `DenseAttention` (D0p/D1/Q0) registers
   the same length-H `omega_h` buffer (D0/standard-RoPE deliberately
   does not); `phase4_model.py`'s TBD `LADDER_BASE = 0.02` removed. γ
   confirmed init-1.0, unconstrained, 1-D → no weight decay. Verified:
   full suite green (P4M1–M7, P4T1–T8, model wiring), grid preflight
   [P1]–[P5] passes at debug dims ([P4] finds and verifies the ladder,
   S == D0p identity), H=6 values match both the grid formula and the
   spec §2 worked table, 20-step dial trainer smoke clean.
4. ~~Tune `GRID_DIMS`; audit the K3 FLOP accounting.~~ **Done
   2026-07-22.** Dims tuned against real `Phase4Model` counts, D0p @
   mlp 1536 (the Phase 3 B0 lineage width) = 13,784,064 params as
   reference: D0 1536 (+0.00%), S/X 1824 (+0.01%), Q0 1727 (−0.01%),
   D1 1992 (params free at +15.3%, the B1-analogue fair fight). FLOP
   formula audited op-by-op against the actual modules — the chat-side
   estimate had placeholder d→d projections (K3's wq/wk are really
   d→H·16) and was missing the H factor on the score-path terms; the
   audited formula's terms are mapped to code in its docstring. With
   it, D1 mlp 1992 matches S FLOPs *exactly* (30,941,184/token), and
   the audited S-vs-D0p compute ratio (1.157×) is consistent with the
   measured 1.17–1.46× wall-clock overhead. Both sentinels flipped
   (`DIMS_TUNED`/`FLOPS_AUDITED`); `--match_table` passes with all
   asserts armed; header checklist items [1]–[4] marked done in-file
   ([5], the `--debug` run, is step 5 below).
5. ~~`--debug` + `--sim_preempt_step 6` resume test.~~ **Done
   2026-07-22, both clean.** `--debug` ran the full path end-to-end
   for the first time (preflight → S + D0p at debug dims → evals +
   len-gen ppl@512/1024 → summaries → grand summary, frozen readout
   correctly deferred on a partial grid; exit 0). Preempt test: exit 3
   fired after 6 steps as designed; the re-run (same command minus the
   flag, exactly the Colab-reconnect workflow) resumed from the last
   eval-boundary checkpoint and every subsequent eval line came out
   **bit-identical** to the uninterrupted run — final val 8.4669,
   k1 spreads, and len-gen ppl all exact, on both the resumed S and
   the fresh D0p. Two timing notes for the GPU spend: checkpoints are
   written at eval boundaries (a disconnect loses at most one eval
   interval of steps), and S ran ~4.4× slower than D0p at the tiny
   debug dims on CPU (einsum overhead at toy sizes — not
   representative of the audited 1.157× compute ratio at grid dims).
6. **The GPU spend — LAUNCHED 2026-07-22** (Colab Pro, A100-SXM4-40GB).
   `PHASE4_colab_launch_log_2026-07-22.md` has the launch narrative,
   incident log (§2), and recovery drill (§4) — still accurate.
   **`PHASE4_session_close_2026-07-22.md` supersedes it for results**
   (its own stated scope: written mid-X at seed 1337, now stale). Drive
   `p4_runs_ts/` is the live source of truth. **12 of 18 runs complete
   — seeds 1337 and 1338, all six variants each, all `diverged=false`.**
   Seed 1339 stopped after D0p reached step 15260/18311 when Colab
   compute units hit zero; the owner is waiting for the free-tier reset
   rather than buying units — **deliberate, not blocked.**
   **Completeness gate holds — no grading, no margins, no verdict
   language until 18/18**; the grand summary fires the frozen §5
   readout itself. Twelve-run table (val / ppl@512 / ppl@1024, seed
   1337 → 1338):
   - D1  1.6367/8.67/24.63 → 1.6428/14.46/35.06
   - D0  1.6404/7.38/18.68 → 1.6463/7.82/19.46
   - D0p 1.6644/12.55/35.05 → 1.6670/6.89/24.41
   - X   1.7638/18.67/52.22 → 1.7640/18.05/51.60
   - S   1.7979/12.14/39.81 → 1.8014/11.22/34.38
   - Q0  1.8241/21.08/45.50 → 1.8271/20.96/47.18

   Descriptive, two-seed, ungraded observations (session-close §3):
   - **(a)** In-distribution ordering D1<D0<D0p<X<S<Q0 is stable across
     both seeds (largest val drift 0.006). S trails D1 (H4a clause 1's
     comparator) by ~0.16 both seeds. Q0 finishing *below* both K3
     variants means attention is doing real work on this task.
   - **(b) The most robust cross-seed pattern in the data:** S beats X
     at ppl@512 on both seeds (12.14 vs 18.67; 11.22 vs 18.05 — margins
     6.5/6.8), while X wins in-distribution by ~0.037 both seeds. The
     scrambled tensor fits better; the true tensor extrapolates better
     at the first context doubling.
   - **(c) Confound for any writeup: the frozen ladder itself costs
     extrapolation**, and inconsistently across seeds — D0 (standard
     32-freq RoPE) vs D0p (frozen 6-freq ladder), same width: ppl@512
     7.38 vs 12.55 (1337, large penalty) but 7.82 vs 6.89 (1338, no
     penalty, D0p even ahead). S and X both carry the ladder, so the
     defensible comparator for S is D0p, **never D0** — exactly why the
     spec made D0 report-only. Do not let an S-vs-D0 comparison into
     any draft.
   - **(d) ppl@1024 is high-variance across seeds (D0p 35.05→24.41, D1
     24.63→35.06 — 30–40% swings on identical configs); ppl@512 is
     comparatively tight** (S 12.14→11.22, X 18.67→18.05). With n=2 the
     512 tightness can't be distinguished from luck, but H4a clause 2
     reads ppl@1024 against D0p — the noisier rung — where S loses on
     both seeds.
   - **(e) Mechanism — the strongest finding of the session, from
     `eval_log.jsonl`'s per-layer r² diagnostics:** every S layer drives
     min(r²) down 8–100× over training (L0: 0.153→0.0013 seed 1337,
     99–117× both seeds); no X layer moves min(r²) more than ~2×, and
     **X never places a single pair below r²=1e-2 in either seed.**
     Since score = −γ·r², the manifold is the attend-here set: S
     inhabits it over training, X does not. S's L0 shows p5(r²) dropping
     in 5/6 heads with low learned γ (~1.2–1.7); deeper layers (L3–L5)
     show p5 *rising* with high γ (~3.1–3.7) in both variants — X only
     ever shows this second regime, never the manifold-seeking one. γ
     never flipped sign in any of 72 head-slots across the four runs.
   - **(f) Wall-clock, recorded not graded:** S/X ≈433s/eval interval
     vs dense ≈181s ⇒ ~2.4×, exceeding the post-hoc 2× bound the grand
     summary will flag. 1.157× is the audited FLOP-invariant cost; the
     residual is this einsum implementation on this hardware, TF32 off
     — a systems question, per the owner's endorsed framing, not a
     property of the math.

   **Two structural findings about the pre-registration itself**
   (session-close §4 — both belong in `RESULTS_phase4.md` as
   methodology notes once 18/18 allows writing it; recorded here so
   they aren't lost before then, per the standing "no RESULTS_phase4.md
   yet" rule): (1) the replicated S-vs-X effect (b) lives at ppl@512,
   a rung H4a's clause 2 doesn't grade (it reads ppl@1024, where S
   loses). (2) H4c's frozen margin (2× the across-seed std of step-0
   p5, and that std is tiny, ~0.004, since init statistics are stable)
   may not discriminate S from X — several X L0 heads decrease by more
   than that margin in both seeds too, so H4c could PASS for both while
   the genuinely discriminating evidence (the 10–100× r²-magnitude gap
   in (e)) sits outside the frozen rule entirely. **Do not cite "H4c
   passed" as evidence of ZD-specificity** if this comes to pass —
   report the magnitude comparison and label it post-hoc. Both findings
   are the freeze working as intended: it committed before anyone knew
   where to look.

   **New check run 2026-07-22, repo-side, no GPU** (session-close §5
   item 1, flagged as the single highest-value cheap item available):
   does R_8's positional decoupling — proven for the TRUE tensor in
   `phase4_positional.py` (checks A–D) — carry over to X? **No, on all
   three counts, all three grid seeds** (`phase4_X_positional_check.py`,
   new, rerunnable): (i) `L_8² = −I` FAILS for X's own shuffled L_8
   (deviation exactly 1 or 2, seed-dependent — not a rotation group);
   (ii) norm preservation is genuinely violated (`|R(θ)v|` drifts from
   `|v|` by up to 0.44 across the frozen ladder × position range, not
   floating-point noise); (iii) shared-phase null preservation is
   BROKEN for every X null pair at every frozen-ladder head, both seeds
   with pairs — confirmed exactly null at t=0 then order-1 violations
   (~1.5–2.9, comparable to the null vectors' own norm √2) within a few
   positions (traced pair-by-pair in the script's docstring). **Status:
   descriptive, NOT applied to the spec, NOT a grading input** — same
   status the Reggiani smoothness certificate held before §9.1 folded
   it in. Sharpens, does not reverse, the existing X caveat: X isn't
   just differently-structured nulls, its own positional rotation isn't
   even unitary and doesn't preserve its own nulls under phase shift —
   position and (shuffled) structure do NOT decouple for X the way
   they're proven to for the true tensor. **Candidate, not asserted,
   mechanistic link to (e):** a manifold that isn't phase-stable under
   the model's own positional code is a harder attend-here target to
   learn than one that provably is — flagged for the eventual writeup,
   held to the same completeness-gate discipline as everything else
   until 18/18. Session-close §5 items 2 (forward-citation traversal)
   and 3 (layer-0 ablation design, needs pre-registration) were **not**
   started — items for an explicit ask, not done proactively (item 3
   in particular touches the frozen spec and needs owner sign-off same
   as any amendment).

   Operational corrections learned at launch, now canonical:
   - Colab upload set is **FOUR files**: `sedenion_kernel.py` (imported
     at module level by `phase4_layers.py`), `phase4_layers.py`,
     `phase4_model.py`, `phase4_grid.py`. The three-file list
     previously here and in the grid's own quick-start docstring was
     wrong; both corrected 2026-07-22.
   - **Two GPU hotfixes synced back to the repo 2026-07-22** (log
     §2–§3): torch.quantile's q tensor was built on CPU against CUDA
     input, and torch.quantile has a hard 2^24-element cap that the
     pooled r² tensor exceeds at grid dims (~25.2M). Pooled p5/med are
     now sort-based nearest-rank (descriptive-only fields, ≤1 order
     statistic vs interpolation; **the graded per-head p5 was never
     affected by either bug**). Applied to `phase4_grid.py` AND
     `phase4_train.py` (which carried bug #1 verbatim and would hit
     bug #2 at grid dims), diagnose() blocks textually aligned;
     `--debug` + trainer-smoke regression rerun clean. Open caveat:
     the sync was reconstructed from the log's documented diffs (this
     machine's Drive mount has no `p4_code/`) — byte-diff repo vs
     Drive at the next Colab session. Also: the recovery stack's
     cell-4 grep fingerprint (`device=r2.device`, "must print 1")
     looks stale post-fix-2, which replaced the line that string was
     on — after the byte-diff, repoint it at a fix-2 marker (e.g.
     `grep -c "nearest-rank"`).
   - Disconnect drill = the six-cell recovery stack (log §4). Always
     `force_remount=True` — a recycled VM can report a zombie Drive
     mount. Launch command identical every time; resume automatic.
   - Drive eventual consistency: cross-VM mid-run log reads are
     unreliable until file close (0-byte reads with fresh mtime); the
     checkpoint-loss bound weakens to "one eval interval beyond what
     had fully propagated" on worst-case VM death. Atomic writes still
     preclude corruption — staleness only.
   - Wall-clock actuals: dense ~36 min/run, S/X ~87 min ≈ **2.4×**,
     exceeding the 2× post-hoc bound the grand summary will flag.
     Decomposition for the writeup: 1.157× is the audited
     FLOP-invariant cost; the residual is this einsum implementation
     on this hardware (many small 16-dim contractions, TF32 off) — a
     systems question, not a property of the math. Not a grading input.
   - Replication watch list (log §5/A5 — pre-registered curiosity, NOT
     new hypotheses): the single-seed 512-ppl crossover (S beat D0p at
     512, lost at 1024); the X-ahead-of-S mid-run ordering; the
     k1-at-init separation (S 1.34 / X 2.70 / dense ~2.6–2.8) as a
     cheap tensor-fingerprint diagnostic.
7. Post-grid, compute permitting: K4 arm, then the entmax arm — which
   FIRST needs the vendored entmax-1.5 + P4M8 NumPy mirror + P4T9
   torch tests per spec §3 (not yet written anywhere).

## 4a. Pre-registration amendments A1–A3, frozen 2026-07-23 (no git repo — this entry is the audit trail)

Delivered by a parallel claude.ai session as three draft files
(`P4_AMENDMENT_A1_magnitude_statistic.md`, `P4_AMENDMENT_A2_rung_posture.md`,
`P4_AMENDMENT_A3_layer0_ablation.md`) plus a routing doc
(`P4_AMENDMENTS_HANDOFF.md`), explicitly requiring verification and
freezing **before any seed-1339 run starts**, to preserve the property
that they were specified blind to a third of the grid. **This project
has no git repo** (`git rev-parse --is-inside-work-tree` fails), so
"commit hash" in each file's own freeze procedure is satisfied by this
log entry plus each file's own dated status header instead.

**What Claude Code verified, all against local files only** (raw
`eval_log.jsonl`/`summary.json` are Colab-Drive-only, not synced to this
repo — the same limitation `PHASE4_session_close_2026-07-22.md` §6 item
A1 already flags for `phase4_grid.py`'s own hotfix sync):

- **A1.1 slot filled** from `phase4_layers.py` directly (not
  `phase4_grid.py`, which imports rather than bundles — the amendment
  assumed the `zda_grid.py` pattern, corrected). Confirmed by code read:
  r² is computed on the **ladder-rotated** q/k (`R_8(ω_h·pos)` applied
  before the sedenion product), not raw projections; `s = -γ_h·r²`
  exact.
- **A1.4's four bars**: arithmetic (the `log10` ratios and separations)
  recomputed independently and matches exactly. Source r² values checked
  against `PHASE4_session_close_2026-07-22.md` §3(e) (match exactly) —
  **not** checked against raw `eval_log.jsonl` (unavailable locally);
  flagged as owed, not silently treated as complete.
- **A2.3**: S-vs-X margins at both rungs recomputed from session-close
  §2's table, match exactly (6.53/6.83 @512, 12.41/17.22 @1024) — the
  control comparison replicates at both rungs with larger absolute
  margins at 1024, confirmed.
- **A2.4**: the proposed correction to session-close §3(d) is **real,
  not just plausible** — independently recomputed relative (max/min)
  spread on the dense family (D0/D0p/D1) gives 82% @512 vs 44% @1024,
  the *opposite* of §3(d)'s original "1024 is noisy, 512 is evidential"
  claim. The likely cause (§3(d) compared absolute deltas, which are
  larger at 1024 purely because the values are 2–3× bigger there) also
  checks out arithmetically. **`PHASE4_session_close_2026-07-22.md` §3(d)
  has been corrected in place**, struck through and flagged rather than
  silently rewritten (this project's standing correction convention).
  Direction: this makes the H4a clause 2 negative *cleaner* (removes an
  unsupported basis for discounting it), not weaker.
- **A3's γ assumption**: confirmed by direct code read of
  `phase4_layers.py` — γ is per-head, and γ_h=0 nulls that head's score
  to exactly 0 (not approximately). **Found and corrected a real error**
  in A3.1's stated rationale: it invoked the T4 (β=0≡standard-attention)
  analogy, but `CLAUDE.md`'s own Phase-4 invariants rule that out
  explicitly for K3 ("no init-time equivalence to standard attention...
  don't try to resurrect a β=0-style baseline-equivalence test"). K3 has
  no score term besides `-γ·r²`, so γ_h=0 doesn't preserve a
  content-based attention pattern underneath — it produces **uniform
  attention over the causal window** for that head (value slice and
  output projection still live, which is what makes it more surgical
  than deleting the head — just not for the reason originally given).
  The intervention itself is unaffected and sound; only the prose
  justification and the "what does a γ=0 head do" characterization were
  corrected. A3's harness (A3.2–A3.3) is **not yet built** — this freeze
  covers the design document only, consistent with the amendment's own
  separation of "design frozen now" from "build/validate/execute later,
  report only at 18/18."

**Open, unresolved by Claude Code, needs the owner:** whether seed 1339
had already started by the time this verification pass ran. Claude Code
cannot observe Colab state directly and the amendments were delivered
before this verification began. If 1339 was already running or done,
all three amendments lose their "specified blind" property per their
own rule and should be re-labelled ordinary post-hoc analysis rather
than presented as blind — a fact-of-timing question only the owner can
answer, not a judgment call to resolve unilaterally.

**Also parked in the routing doc** (`P4_AMENDMENTS_HANDOFF.md` §6),
listed there as not yet merged: `PRIOR_ART_REVIEW_zda_section8-4_draft.md`.
**This is now stale** — that draft was independently delivered,
verified, and merged into `PRIOR_ART_REVIEW_zda.md` §8.4 earlier the
same day, in a session that ran before this one and didn't yet know
about the amendments. No action needed; noted here so the discrepancy
doesn't cause confusion later.

## 4b. I1–I4 executed 2026-07-23/24 (session ended on a "freeze and save" warning — see status flags below)

Routing doc: `P4_AMENDMENTS_HANDOFF.md` §I1–I4 (superseded mid-session by
a second doc, `SESSION_CONTINUATION_HANDOFF_2026-07-23.md`, dropped into
the repo root by the parallel chat session — **read that file too**, it
has independent context this section doesn't repeat, including the
owner's decision to buy Colab units for seed 1339 rather than wait for
free-tier reset).

- **I1 (seed 1339 timing) — UNRESOLVED, blocking.** Claude Code cannot
  check Colab billing history or Drive `p4_runs_ts/` mtimes from this
  machine — no access to either. Per the routing doc's own rule ("take
  the weaker label under any uncertainty"), **A1, A2, A3, and
  A3-revised should all be treated as ordinary post-hoc analysis, not
  presented as blind, until the owner confirms otherwise.** This is
  the single most important open item for the next session to close
  with the owner directly — a one-question ask, not investigation.
- **I2 (git init) — DONE.** No git repo existed in this project before
  this session. Initialized, `.gitignore` added (excludes `zda-env/`
  venv [22k files], `data/`, `runs*/` [checkpoints], `.claude/settings.local.json`).
  Initial commit `fe2e967` (2026-07-23 16:11:25 -0700, 63 files) covers
  the repo state including the already-frozen/corrected A1/A2/A3 and
  the merged §8.4. A second commit added `P4_AMENDMENT_A3_REVISED_layer0_dose_response.md`
  and `phase4_shared_phase_autotopy_check.py` (hash TBD — see below,
  commit may not have completed before session end; check `git log`
  first thing next session).
- **I3 (A3 reopened) — DONE.** Verified empirically (script run, not
  just code read — `gamma[h]=0` gives *exactly* uniform, content-blind
  attention for that head, zero deviation, other heads bit-identical).
  This voids A3.1's original selection argument (it claimed γ=0 "leaves
  content-based attention intact," which is false for K3 — no
  dot-product term survives underneath, unlike Phase 1–3's β=0). The
  intervention itself was still sound; only its justification was wrong.
  **`P4_AMENDMENT_A3_REVISED_layer0_dose_response.md` frozen** —
  supersedes (does not edit) the original A3: a γ-multiplier
  dose-response sweep (m ∈ {0, 0.25, 0.5, 1.0, 2.0} on L0, both S and X,
  both rungs, 3 seeds) plus a clamp-to-deep-layer-γ arm, replacing the
  single binary ablation. One ambiguity in the proposed design (whether
  the clamp arm's "deep-layer median" is per-head or pooled) resolved as
  **pooled** (no evidence head identity is consistent across layers),
  stated explicitly so it's fixed before, not after, execution. Harness
  still not built — design freeze only, same division as the original.
- **I4 (shared-phase null / autotopy check) — DONE, then partially
  superseded in priority.** Prerequisite question answered by direct
  code read (no computation needed): `shuffled_structure_tensor` draws
  **15 mutually independent permutations** (one fresh `g.permutation(16)`
  per k=1..15), not one consistent global relabeling — **X is an
  arbitrary bilinear map, not an isomorphic copy of the sedenions in a
  misaligned basis.** The parallel chat session reached the identical
  conclusion independently (see `SESSION_CONTINUATION_HANDOFF_2026-07-23.md`
  §3) and flagged its consequence as **higher priority than the
  autotopy check**: if X has no accessible zero-divisor variety at all,
  "X never reaches r²<1e-2" (session-close §3(e), used throughout this
  session's other write-ups) may be vacuous rather than a real finding
  about training — the S-vs-X positive result would need reframing from
  "coherent algebra beats broken map" toward something closer to the
  Phase 1–3 V1/V4 confound (weight-sharing/structure, not algebra).
  **`phase4_shared_phase_autotopy_check.py` run and committed** — true
  T16's induced autotopy defect vanishes strictly faster than X's as
  the rotation angle shrinks (ratio grows from ~1.3–1.5× at the largest
  frozen frequency to ~45–53× at the smallest, all 3 seeds) — a real,
  quantified finding, but **the chat session's proposed follow-up (does
  X have zero divisors at all — minimize ‖x⊗y‖² over unit x,y, report
  the infimum) is more decisive and was started at session end
  (`phase4_X_variety_characterization.py`, background run, result
  status:** — **RESOLVED before session end, result is IN:** found
  **true T16's infimum = 2.0e-8; X's infima = 3.7e-8 (1337), 1.2e-8
  (1338), 2.5e-8 (1339)** — all at the same optimizer noise floor, no
  meaningful separation. **X has its own accessible near-zero-divisor
  region, comparable to the true tensor's.** This refutes the "X may be
  vacuous" worry: "X never reaches r²<1e-2 during training"
  (session-close §3(e)) is a real claim about what training did or
  didn't find, not an artifact of X having nowhere to descend to.
  **The S-vs-X positive result survives this check** — its
  interpretation should be refined, not discarded: X has zero-divisor-
  *like* structure somewhere in its space, just apparently not one
  training discovers/exploits the way it does for the true tensor's
  actual manifold — consistent with, and now better supported by, the
  existing `phase4_X_positional_check.py` finding that R_8's positional
  decoupling fails for X. **Not yet folded into
  `PHASE4_session_close_2026-07-22.md` §3(e) or any amendment's prose**
  — do that next session, flagged the same way the §3(d) correction was
  (struck through, dated, reasoned), not silently.

**Everything above is descriptive and NOT applied to the spec or any
grading — completeness gate (no verdict language until 18/18) is
untouched by any of this.** The X-variety question, if it comes back
"bounded away from zero," would be a significant enough finding to need
explicit owner attention before any further write-up leans on the S-vs-X
framing as currently stated in `PHASE4_session_close_2026-07-22.md`
§3(e) — flag it, don't quietly rewrite that section without a similar
struck-through-and-flagged treatment to the §3(d) correction.

## 4c. Reply received 2026-07-24 — X-variety thread closed, §3(e) addendum applied, A1 defended

`REPLY_to_ClaudeCode_2026-07-24.md` + `FINDINGS_zd_variety_characterization_2026-07-24.md`
(same parallel chat session) responded to §4b. Their contribution:
reconciled the two independent X-variety runs (their σ_min/SVD floors at
~1e-19; Claude Code's squared-GD floors at √ε_f64≈1.49e-8 — arithmetic
checked, they agree, both are "zero" not "small positive"), then raised
two sharper points that needed real-artifact verification, not just
class-level reasoning (their session had no repo access): (1) X's
14-dim ZD variety is **generic** to any bilinear map, not sedenion-
specific — doesn't distinguish X from noise; (2) under free-sphere
sampling, **X's variety is ~440–465× *more* accessible than S's**
(median σ_min: S 0.478, X 0.036–0.040) — the reverse of what "S seeks
the manifold, X doesn't" would suggest, raising a baseline-asymmetry
confound against A1's fold-change statistic and an unreconciled
discrepancy against session-close §3(e)'s "X never crosses r²<1e-2."

**Claude Code verified both against the real repo artifacts (not the
guessed shuffle-spec readings the chat session used) and ran the
decisive follow-up (N2) themselves:**

- `phase4_X_variety_sigma_min.py` — confirms the accessibility numbers
  **exactly** on the real seeds 1337/1338/1339 (median X/True frac<1e-1
  ratio 493–497×, method validated against quaternions/octonions giving
  infimum=1.000000 exactly). The chat side's analysis holds up.
- `phase4_init_r2_check.py` (**N2, the decisive check**) — simulated
  fresh (untrained) instances of the *real* grid architecture
  (d_model=384, 6 heads, ctx=256) and read off the actual r² the model
  computes at init, rather than free-sphere x,y. **Result: the model's
  real q,k pairs do not resemble free-sphere samples at all.** X's
  simulated init r² distribution (median ≈0.97, 0 of a large batch below
  1e-2) looks like S's, not like the free-sphere prediction (median
  ≈0.04, ~13% below 1e-2) — and both variants' simulated step-0 minima
  (S 0.121–0.173, X 0.044–0.092) closely bracket the real grid's own
  recorded step-0 numbers (S 0.1529/0.1382, X 0.0522/0.0820), a strong
  cross-check that the simulation is realistic. **This resolves both
  open questions**, and mostly in the original finding's favor: the real
  step-0 gap between S and X is only ~2–3×, far too small to explain the
  training-driven gap (S descends 99–117×, X 1.04–2.2×) as a baseline
  artifact. The "X may be vacuous" and "baseline-asymmetry confounds D"
  worries are both substantially refuted for the real model, though two
  refinements to the framing were adopted anyway (attribute the finding
  to what training locates/exploits under the real parameterization, not
  to abstract zero-divisor availability, since X's is *more* accessible
  in the abstract).

**Applied:** `PHASE4_session_close_2026-07-22.md` §3(e) got a substantial
addendum (not a strike-through — the original numbers are accurate, only
the interpretation needed qualifying) with the full chain above and
citations to both new scripts. `P4_AMENDMENT_A1_magnitude_statistic.md`
got a short addendum noting the baseline-asymmetry challenge was raised
and checked, and D's cross-variant comparison held up. **No change was
needed to A3-revised's premise** — the chat side's suggestion to hold it
pending N2 is satisfied; N2 returned and didn't undermine the
manifold-seeking premise.

**Not yet done:** N3 (pull existing step-0 r² straight from
`eval_log.jsonl` rather than this session's simulated proxy) remains
blocked — raw logs are Colab-Drive-only, same limitation as everywhere
else in this document. The process flag in `REPLY_to_ClaudeCode_2026-07-24.md`
§5 (two chat sessions running in parallel against one repo without a
lock) is worth the owner's attention but is a coordination question, not
something Claude Code can resolve unilaterally.

**I1 (seed 1339 timing) is still the top open item, unchanged by any of
this thread.**

## 4d. I1 RESOLVED 2026-07-24 — A1/A2/A3/A3-revised labeled POST-HOC, NOT BLIND (corrected)

Owner pasted the seed-1339 Colab log directly (D0p stopping at step
15260/18311, same point as the 2026-07-22 quota exhaustion this
document already recorded) and, asked directly, **confirmed it is the
same historical 2026-07-22 output, not a new run since.**

**Claude Code's first pass at this drew the wrong conclusion from that
fact — corrected by Claude Desktop before the erroneous version was
committed (the commit was rejected, nothing wrong landed in git).** The
error: testing "did new seed-1339 compute happen *after* the amendments
were frozen" and concluding blind, since the answer was no. **The
correct test is whether seed-1339 data existed *before drafting*, not
whether it changed *after freezing*.** The amendments were drafted
2026-07-23; the partial D0p/1339 run (train/val loss through step 15260,
no `done:` line, so no len-gen ppl@512/1024) is dated **2026-07-22 —
one day before drafting.** Pre-registration blindness requires the data
not to exist, or not be observed, at the time of specification; data
that already existed the day before, sitting in the same
`PHASE4_session_close_2026-07-22.md` the amendments' own bars were set
from, forecloses that regardless of whether anyone actually looked at
it. **The owner was asked directly and does not recall** whether that
partial output was consulted before drafting — a genuine "don't know,"
not a "wasn't asked." Per the standing rule ("take the weaker label
under any uncertainty"), that uncertainty resolves to the weaker label
rather than being left open: **the correct label is post-hoc, not
blind, for all four amendments (A1, A2, A3, A3-revised) uniformly.** Each file's "Ordering
caveat" section corrected (struck through, not deleted, twice now —
once for the original open caveat, once for Claude Code's own
mislabeling — both kept for the record, per this project's standing
correction convention of flagging errors rather than erasing them).

**This does not remove the amendments' value** — a statistic fixed
before seeing the *full* 18-run picture still beats one fixed
afterward — it only requires accurate labeling in any future write-up:
call it post-hoc analysis with early specification, not a
pre-registered-blind test.

Owner's choice for next steps on seed 1339 itself: **keep retrying on
Colab A100** (not switching to T4, not restructuring into smaller
chunks, not investigating the step-15260/18311-ish stall point
specifically — those were offered as options and declined).

This closes the single item that has been open across the entire
multi-session amendment thread (§4a → §4b → §4c → this), with the
correct answer rather than the first, wrong one. Nothing else in this
document changes as a result — completeness gate (no grading until
18/18) is untouched, and 1339 itself is still incomplete.

## 4e. GRID COMPLETE 18/18 (2026-07-25) — and an H4c readout bug found and fixed before any verdict is trusted

**The grid finished.** All 18 runs (6 variants × 3 seeds) are done. This
is the milestone the completeness gate has been protecting since
2026-07-22. **But the first grand-summary printout is not to be trusted
for H4c** — see below. H4a/H4b′ are unaffected; do not let the H4c
incident cast doubt on those.

**The incident.** The 18/18 grand summary printed H4c (S) as
"negative," with **every per-head margin shown as `nan`** and numpy
warnings ("Degrees of freedom <= 0 for slice", "invalid value
encountered in divide"). A "negative" verdict computed from `nan`
margins is not a genuine result — in IEEE754/numpy, any comparison
against `nan` returns `False`, so a fully-nan margin array makes
`hits` empty and prints "negative" **without ever actually testing a
single head.** This was flagged immediately as unadjudicated, not
negative, before being investigated.

**Root cause, confirmed by direct code trace (not guessed):**
`phase4_grid.py`'s `h4c_readout()` looped `for seed in seeds`, trusting
the `seeds` *parameter* it received from `grand_summary()`, which in
turn came straight from `main()`'s `--seeds` CLI argument — the same
argument that also controls which seeds that particular invocation's
**training** loop iterates. Those are two different concerns that got
conflated: a later Colab invocation narrowing `--seeds` to just the one
seed that still needed finishing (very plausible here, since seeds
1337/1338 already had `summary.json` on disk and would be skipped by
the `os.path.exists` check regardless of what `--seeds` said) silently
propagated into the **grading readout** too. With `seeds=[1339]`,
`h4c_readout`'s `p0` array had a seed axis of length 1; `p0.std(axis=0,
ddof=1)` on a length-1 axis divides by zero (`N - ddof = 1 - 1 = 0`),
producing exactly the observed warnings and `nan`. **Confirmed this
isn't a misreading of the code**: `phase4_grid.py`'s local sha256
(`65d419e05b6ec74dbfd2449c2548974290f668ca59be2618e78c803883a748ab`)
matches the hash of the script that actually produced all 18 runs,
checked before trusting the code-read at all. H4a/H4b′ were unaffected
because they aggregate via `glob.glob(out_root/*/summary.json)`
(auto-discovering every run on disk), never via the `seeds` parameter —
this asymmetry is exactly why only H4c broke. This is also a direct
violation of spec §5's own explicit rule, **"No grading from partial
seeds"** — the bug is that the code didn't enforce a rule the spec
already stated.

**Fix applied to `phase4_grid.py`** (`h4c_readout`, plus a new
`EXPECTED_N_SEEDS = 3` constant): seeds are now **discovered from the
run directories actually present on disk** for the variant being
graded (`glob` on `{variant}_seed*`), the same pattern H4a already used
safely — never enumerated from the `seeds` argument, which is now used
only as an informational cross-check. If the discovered seed count
isn't exactly 3, the function refuses to grade ("cannot grade... No
grading from partial seeds") and returns `None`, rather than silently
computing on whatever subset happens to be present. **Separately, a
generalizable robustness guard was added**: if a margin is ever
non-finite even after that gate passes, the function now **raises**
rather than lets a downstream comparison silently fall through to
`False`. A verdict must never be emitted from `nan`.

**Verified, not just written:** no access to the real 18-run
`eval_log.jsonl`/`summary.json` exists in this repo (Colab-Drive-only,
same limitation as every other data-dependent check this project has
hit) — so the fix was verified with a **synthetic** reproduction
(`test_h4c_fix.py`, run this session, not committed — scratch-only)
matching the real data schema exactly: (A) 3 complete synthetic seeds
compute real, finite margins with zero `RuntimeWarning`s (checked by
promoting warnings to errors) and return a proper bool; (B) the *exact*
bug scenario — only one seed's directory present, called the way a
narrowed `--seeds` invocation would — now correctly prints "cannot
grade" and returns `None`, not a `nan`-driven false "negative". Both
checks passed. The existing `--debug` end-to-end path was also launched
as a supplementary regression check; `import phase4_grid` succeeding at
all (required for the synthetic test to run) already rules out any
syntax-level breakage from this change.

**What this fix does NOT do, per the task's explicit constraints:**
it does not change H4c's statistic, margin, or threshold — spec §5's
definition is untouched. It does not re-train anything.

**Update, same day: the real data landed, and the corrected verdict is
in.** The owner synced the actual grid outputs to `p4_artifacts/`
(all 6 variants × 3 seeds, `eval_log.jsonl`/`summary.json`/
`train_log.csv`, 56 files, committed separately at `e017bc2` — data
only, no code in that commit). This is the first time this repo has had
direct access to the real grid artifacts rather than working from
hand-written descriptive summaries. Ran the corrected `h4c_readout`
directly against it, then the full `grand_summary()` for the coherent
picture (`phase4_grid.py`, unmodified since the `415506e` fix — no
further code changes to produce this readout):

```
================ GRAND SUMMARY ================
variant   n    val loss (mean ± std)   ppl@512  ppl@1024    k1 min  wall h
D0p       3         1.6620 ± 0.0066      9.84     28.90  2.54e+00    0.60
D0        3         1.6400 ± 0.0065      7.89     20.69  2.54e+00    0.60
D1        3         1.6381 ± 0.0042     13.34     34.04  2.53e+00    0.67
S         3         1.7980 ± 0.0033     11.37     34.90  1.33e+00    1.44
X         3         1.7632 ± 0.0012     18.22     51.64  2.42e+00    1.44
Q0        3         1.8211 ± 0.0079     20.54     45.88  0.00e+00    0.54

H4a-1: S - D1 val = +0.1600, margin 0.0076 -> negative
H4a-2: S - D0p ppl@1024 = +6.001 (margin 10.216) with |S - D0p| val =
       0.1360 (within 0.0104) -> negative
H4a: NEGATIVE
H4b': not evaluated (conditional on an H4a win)
H4c (S, per-head p5(r2) step-0 vs final, margin 2x pooled step-0 std,
     same head in all seeds): PASS — heads
     [(0,1), (0,2), (0,3), (0,4), (0,5), (2,3)]
X manifold trace (descriptive, NOT graded): heads [(1,1), (2,0)] moved
     beyond the H4c-style margin
Wall-clock S/D0p = 2.39x (EXCEEDS the 2x bound)
```

**H4c (S) is a genuine, adjudicated PASS** — five of L0's six heads
(H1–H5) plus one L2 head (H3) cross the frozen margin in all 3 seeds,
consistent with the two-seed descriptive read's "S's L0 shows five of
six heads with p5 dropping" (session-close §3(e)), now confirmed graded
and with a third seed and one additional deeper-layer head added. **This
answers the original task's steps 5–6 directly: the verdict changed
from "unadjudicated (nan fall-through)" to a real, code-computed PASS —
not assumed, not guessed.**

**One correction this surfaces that needs to be carried forward
carefully**: the "X manifold trace" descriptive line is the *same*
`h4c_readout()` function called with `graded=False` — it was **equally
affected by the nan bug** (the task's own evidence already showed "same
nan pattern on all five rows" for X too). The corrected version shows
**X does have 2 heads crossing** (L1H1, L2H0) — smaller drops, different
layers than S's L0-heavy pattern, and far fewer heads (2 vs 6) — where
the buggy readout had (falsely) shown zero. **This is a different
statistic from session-close §3(e)'s "X never places a pair below
r²=1e-2"** (that's `frac_r2_lt_1e2`, computed independently in the
per-run trainer diagnostics, not through `h4c_readout` at all, and is
unaffected by this bug — it remains accurate as reported). Don't
conflate the two: "X never has a pair below the 1e-2 floor" (still true)
and "no X head ever crosses the H4c-style p5 margin" (was false, an
artifact of the same bug — 2 X heads do cross, modestly).

**Explicitly not done here, and not to be inferred from this entry**:
no interpretation of the H4a-negative/H4c-positive combination has been
drafted. The frozen interpretation table (spec §5) only names
H4a+/H4b′+, H4a+/H4b′−, and all-null — it does not have a named entry
for "H4a negative, H4c positive," and inventing one now would be exactly
the post-hoc narrative drift the table exists to prevent. This needs
deliberate attention (a `RESULTS_phase4.md` drafting session), not an
inline read appended to a bug-postmortem section. **What's safe to say
now**: the mechanism (S measurably migrating toward its own manifold at
several heads, across all 3 seeds) is real and graded-positive; it did
not translate into a value win on this task under the frozen H4a rule.
Both facts stand together and neither cancels the other.

**Provenance for this readout specifically**: produced by importing
`phase4_grid` and calling `h4c_readout()` / `grand_summary()` directly
against `p4_artifacts/` — the same functions, same file
(`phase4_grid.py`, unchanged since commit `415506e`), same frozen rules.
Not a new script, not a reimplementation.

**Provenance, per the task's explicit constraint:** the script that
produced all 18 runs is fixed at sha256
`65d419e05b6ec74dbfd2449c2548974290f668ca59be2618e78c803883a748ab`,
recorded here before any fix commit. All 18 runs were complete before
this fix was written — this is a readout-code fix only, applied after
training, never touching or re-triggering it. Committed as its own
commit (not amended into any prior one), so the boundary between "the
script that ran the grid" and "the script with the corrected readout"
stays unambiguous in git history.

## 5. Loose threads — real, not forgotten, just not urgent

- **Forward-citation traversal (`PRIOR_ART_REVIEW_zda.md` §7 item 3,
  §8.3, 2026-07-23) — addressed, not closed, and the first pass was
  corrected after owner pushback.** Traversing from Zoology/Reggiani
  themselves (174/1 citing papers, all scanned) turned out to be a
  low-power test — that citing community had little reason to use
  hypercomplex vocabulary either way, so the negative was likely a
  priori and doesn't much move the novelty claim; Reggiani's single
  citation gave the sedenion-math branch no traversal power at all. A
  second, better-rooted pass from GATr/Clifford Neural Layers (topic
  survey, Semantic Scholar's API rate-limited this session so this
  wasn't exhaustive pagination) found the whole "geometric algebra
  attention" family — GATr, CliffordNet, CFA, GA point-cloud attention,
  many-body geometric attention — all scoring with the scalar/grade-0
  part of the geometric product, i.e. the same dense-equivalent bilinear
  class our memo §3 already rules out, none using null/zero-divisor
  structure as the score. Still genuinely open: STAResNet
  (indefinite-signature Clifford, not attention, checked only at
  abstract level) and exhaustive structured citation counts for
  GATr/Clifford Neural Layers themselves (API was down). **The
  recommended novelty argument for any writeup is now the positive,
  structural one** (CD-algebra ML stops at the octonions because
  division fails and treats zero divisors as the reason not to go
  further; K3 inverts that boundary), with these searches as scoped
  supporting evidence, not as the proof.
- **§8.4 landed 2026-07-23 (`PRIOR_ART_REVIEW_zda.md`), merged by Claude
  Code after independently re-verifying every load-bearing claim against
  the source paper.** A parallel claude.ai session backward-traversed
  from Reggiani's own `[SA20]` citation — a route forward traversal
  structurally couldn't take (that branch never cites Zoology) — and
  found the live Cayley-Dickson hypercomplex-ML literature (Khalifa
  University sedenion-network lineage; the field's 2024 framework paper,
  arXiv:2405.07024). **The positive framing argument above is no longer
  just an inference**: that framework paper's own Table 1 tabulates
  algebraic properties only through the octonions — no sedenion row, in
  a paper that introduces sedenions two pages earlier — confirmed
  verbatim by Claude Code against the paper's actual text before merge.
  Also sharper novelty statement added: the hypercomplex-attention
  literature (PHAtt, §4.1 of the same paper, confirmed verbatim:
  softmax(QKᵀ/√d_k) on PHM-derived Q/K/V) puts the algebra in the
  *projections* and scores with a plain dot product — K3 puts the
  algebra in the *score itself*. Two Tier-2/3 adjacencies added (Lie-
  Algebra Attention arXiv:2606.20547, Versor arXiv:2602.10195), both
  confirmed real. **Still not claiming closure** — §8.4.7's own open
  items carry forward: PHM (Zhang et al. ICML 2021) forward traversal
  via Semantic Scholar API once reachable, patent literature unexamined,
  STAResNet and exhaustive GATr/Clifford citation counts still pending.
- The G₂-invariance Lean stub in the owner's *other* project
  (`canonical_six_publication/`) — theirs to pick up, Reggiani is the
  likely missing citation, don't touch that repo unasked.

## 6. Things not to re-litigate (already closed doors)

All in `CLAUDE.md`'s invariants sections, worth restating because
they're easy to accidentally reopen:

- K1 (linear collapse score) is degenerate — never build it.
- Variant R (random-frame K3) is vacuous, dropped in spec v0.3 — don't
  re-add it as a "control worth trying."
- Real-part bilinear attention scores can never see ZD structure — a
  closed theorem (memo §3), not an empirical question.
- `R_8(θ) = cosθ·I + sinθ·L_8` is the *unique* compatible positional
  generator, per-head single-frequency — proven in `phase4_positional.py`,
  not a design choice to revisit.
- Phase 1–3's β=0≡baseline anchor does **not** carry over to K3 — K3 is
  mandatory by design, there is no equivalent init-time dense-equivalence
  test to expect or resurrect for it.
- `ZD_PAIR = (e3+e12, e5+e10)`, Baez convention. Don't change without
  logging why (precedent: `RESULTS_phase2_smoke.md` §6).
- γ (Phase 4) and β (Phase 1–3) are never weight-decayed. Same reason
  both times: decaying a 1-D gate/scale param suppresses it by fiat
  rather than by what SGD actually learns.

## 7. Orientation for a fresh model specifically

If you're starting cold: read `CLAUDE.md` in full first (it's dense but
current), then this file, then whichever of `PHASE4_gate5_outcome.md` /
`PHASE4_reggiani_reading.md` / `LIE_ALGEBRA_CONNECTIONS.md` is relevant
to what you're asked to do — don't read all of them speculatively, they're
each self-contained for their own topic. The memory system (if available
to you) has a `MEMORY.md` index with more granular, dated entries than
either `CLAUDE.md` or this file carry — useful for "what exactly happened
and why" on any specific decision, less useful for "what's the state of
the project right now" (that's what this file and `CLAUDE.md` are for).

One adjacent-project note worth internalizing early: the owner has a
separate, much larger research program (Riemann-Hypothesis-adjacent,
tags `#rh-investigation` / `#zdtp` / `#cailculator` in their KSJ journal,
project directory `canonical_six_publication/` and others under
`C:\dev\projects\`) that shares the same sedenion zero-divisor math but
is **not** this project. If a file, term, or tool reference doesn't
resolve inside `apm-agi_tests`, it may belong to that program — check
before assuming it's misplaced or asking the owner to explain from
scratch. See the `canonical-six-publication-reference` memory entry.
