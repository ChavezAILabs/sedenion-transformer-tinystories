# HANDOFF addendum — chat session 2026-07-22 (Claude, claude.ai)

**Append to / merge with `HANDOFF.md`; update `CLAUDE.md`'s status
paragraph to match.** This session ran OUTSIDE the repo (three docs +
`zda_grid.py` + `phase4_train.py` uploaded to chat); everything below
therefore lands in the repo via the three delivered files, none of which
has touched repo state yet.

## What happened (all owner-directed, all owner-approved)

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
3. **Spec frozen and bumped to v1.0** (`ZDA_phase4_spec.md`, delivered):
   all seven diff blocks from `spec_v1_freeze_diffs.md` applied by
   surgical edit to the uploaded v0.3 — untouched text preserved
   verbatim; H4d's own TBDs deliberately survive (retained-verbatim
   rule). Gate 6 marked DONE 2026-07-22.
4. **`phase4_grid.py` built** (delivered) — the anchor-grid orchestrator
   that did not exist. Follows the `zda_grid.py` pattern (match table →
   preflight → sequential variant×seed with atomic resume → grand
   summary ending in the frozen §5 readout). Three documented
   deviations: imports `phase4_layers`/`phase4_model` instead of
   bundling (verbatim-copy rule can't be honored by retyping absent
   files); TF32 hard-disabled on CUDA, no escape flag (spec §9.2); no
   β=0 preflight — replaced with §7 anchors (ZD annihilation, γ
   init/no-decay, K1-guard-alive, causality, ladder verification,
   T=1024 forward). Reuses the Phase 3 `data_ts` cache.

**Test coverage, honestly stated:** no torch existed in the chat
environment. The new decision logic (frozen H4c all-seeds-same-head
rule incl. the 2-of-3-seeds negative case, H4a two-clause + conditional
H4b′ arithmetic, resume-safe last-line-per-step log dedupe, ladder
endpoints) was EXECUTED against synthetic run outputs and passed. The
trainer/preflight/data paths are syntax-checked only.

## Ordered path to launch (each gates the next)

1. Land the three files. Merge this addendum; refresh `CLAUDE.md`.
2. **Residual factual verifications** (spec v1.0 changelog; neither is
   a threshold — a failure is a logged v1.0.1 correction, not a
   re-opening): (a) §9.1 certificate figures (σ_min = √2, 3–9%
   singularities) against `PHASE4_reggiani_reading.md` and the
   certificate script's own output; (b) `phase4_positional.py`
   relative-property spot-check at the six frozen frequencies
   (expected pass — the property is frequency-agnostic per head).
3. **Wire the frozen ladder into `phase4_layers.py`** — identical for
   S/X (R_8 action) and D0p/D1 (per-head single-frequency RoPE),
   exposed as a length-H tensor attribute `omega_h` on the attention
   module; `phase4_grid.py` preflight [P4] refuses to run until it can
   find and verify it. Confirm γ is init-1.0 unconstrained (it should
   already be — the smoke ran that way).
4. **Tune `GRID_DIMS`** mlp widths against real param counts to ±1% of
   D0p; flip `DIMS_TUNED`; iterate with `--match_table`. **Audit the
   K3 FLOP accounting** (the in-file formula is a documented estimate);
   flip `FLOPS_AUDITED` to arm the D1-vs-S ±5% assert.
5. `python phase4_grid.py --debug` and
   `--debug --sim_preempt_step 6` (first real execution of the
   trainer path; CPU, minutes).
6. **The GPU spend** (Colab Pro): upload `phase4_layers.py`,
   `phase4_model.py`, `phase4_grid.py`; `data_ts` cache on Drive if it
   survived Phase 3, else one-time rebuild (~15 min); run the 18-run
   grid; budget for K3's 1.17–1.46× measured overhead with TF32 off.
   Re-run the same command after any disconnect (resume is automatic).
7. Post-grid, compute permitting: K4 arm, then the entmax arm — which
   FIRST needs the vendored entmax-1.5 + P4M8 NumPy mirror + P4T9
   torch tests per spec §3 (not yet written anywhere).

## Unchanged loose threads

Forward-citation traversal from Zoology/Reggiani
(`PRIOR_ART_REVIEW_zda.md` §7 item 3) — still the only open prior-art
item; the G₂-invariance Lean stub in the owner's `canonical_six_
publication/` repo — theirs, other repo, Reggiani likely the missing
citation. Closed doors all still closed: no K1, no variant R, no
β-style equivalence test for K3, no within-head multi-frequency
positional, `ZD_PAIR` unchanged, γ/β never weight-decayed.
