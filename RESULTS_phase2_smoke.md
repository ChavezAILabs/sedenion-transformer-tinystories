# RESULTS — Phase 2 Smoke Run (6 variants, Tiny Shakespeare)

**Date:** 2026-07-17 · **Seed:** 1337 · **Hardware:** Celeron N4120 (4-core, no GPU), torch CPU
**Config (all variants):** 4 layers, 4 heads, ctx 128, 2000 steps, batch 16, AdamW lr 3e-4 cosine (200 warmup), grad clip 1.0, char-level Tiny Shakespeare.

> Per HANDOFF §5 this run exists to catch bugs and confirm sane training behavior,
> **not** to draw conclusions — 1 seed, 2k steps, char-level data. Observations below
> are flagged as patterns to watch in the real grid, nothing more.

## 1. Run integrity

- All 6 variants completed 2000 steps with smoothly decreasing losses; no NaN/instability.
- Step-0 regression check (V2 loss == B0 loss on identical batch/init, the β=0 invariant): **passed**.
- Param matching: B0/V2/V3 at ~799.6k, V1/V4 at 799.7k (within ±1%); B1 FLOP-matched at 2.48M FLOPs/tok vs V1/V4 2.50M (within ±5%).
- Operational note: the run was interrupted twice (session kill at ~step 400 on 07-16 overnight attempt; machine reboot at 04:15 on 07-17 during V3). Because everything is seeded, restarts reproduce trajectories. `run_smoke.ps1` now skips variants with an existing `summary.json`, so re-running it after an interruption resumes at the first incomplete variant. V3/V4 wall times are inflated by post-reboot antivirus contention; losses are unaffected.

## 2. Final losses (step 2000)

| Variant | Role | Params | FLOPs/tok | Train loss | **Val loss** | Wall time |
|---------|------|--------|-----------|------------|--------------|-----------|
| B0 | baseline, param-matched | 799,616 | 1,851,648 | 1.6069 | **1.7467** | 2636 s |
| B1 | baseline, FLOP-matched (d=160) | 1,080,928 | 2,478,400 | 1.5361 | **1.6862** | 3160 s |
| V1 | PHM-16 Q/K/V/O (H1) | 799,692 | 2,504,736 | 1.8628 | **1.9369** | 3182 s |
| V2 | ZD-gate attention (H2, primary) | 799,632 | 1,859,840 | 1.6068 | **1.7468** | 3029 s |
| V3 | gate control: random frozen frame | 799,632 | 1,859,840 | 1.6070 | **1.7465** | 3669 s |
| V4 | structure control: shuffled tensor | 799,692 | 2,504,736 | 1.8420 | **1.9396** | 3444 s |

## 3. β trajectories (V2, V3 — logged per head every 100 steps)

- Both gated variants keep β near zero for the entire run. Growth is slow and roughly monotone in magnitude, no sign flips late in training, no head "switches on."
- **V2 (ZD frame):** final per-head β range −0.011 … +0.021; max |β| = 0.021 (head 3, layer 1).
- **V3 (random frame):** final range −0.016 … +0.042; max |β| = 0.042 (layer 1). The random-frame control actually grew slightly *larger* gates than the ZD frame.
- At |β| ≤ 0.04 the gate term is a negligible perturbation of the attention logits, which is why V2, V3, and B0 val losses agree to ~3–4 decimal places.

## 4. Observations (smoke-level only — patterns to watch, not results)

1. **Gate unused at this scale.** β→~0 everywhere means the model did not choose to use the gate (the honest-reporting case HANDOFF §5 anticipates). Consequently V2 ≈ V3 ≈ B0. If this pattern replicates in the real grid, the spec §1 falsification condition ("if V2 ≈ V3, any gain is not ZD-specific") applies.
2. **V1 ≈ V4, both well below baseline.** PHM-16 is ~0.19 nats worse than B0 in val loss, and the shuffled-tensor control matches it (1.9369 vs 1.9396). Consistent with spec §8 known risk 1 (16× sharing bites hardest at tiny scale) and the expectation that generic structure captures most of PHM's behavior. Not read as H1's death at this scale, per the spec.
3. **Baselines behave as expected.** B1 (more params, FLOP-matched) beats B0 by ~0.06 nats. Nothing anomalous.

## 5. Next step

The real pre-registered grid (18 runs: 6 variants × 3 seeds, TinyStories) is out of scope
for this laptop (HANDOFF §6) — it goes to free Colab/Kaggle GPU. Next concrete action:
prepare a single self-contained notebook/script bundling `zda_layers.py`, `model.py`,
`data.py` (TinyStories loader), `train.py`, and the 6 configs, parameterized by seed.

Artifacts: per-variant `runs/<VARIANT>_seed1337/` — `train_log.csv`, `eval_log.jsonl`
(with per-head β), `summary.json`, `ckpt.pt`. Stale partial logs from the two
interrupted attempts are archived in `runs/_stale_partial_20260717/`.

## 6. Deviation log — ZD frame changed for Phase 3 (2026-07-17)

Before launching the Phase 3 grid, the ZD_PAIR frame init was changed from
`(e1+e10, e4−e15)` (used in Phase 1 validation and this smoke run) to
`(e3+e12, e5+e10)` — KSJ "Pattern 2" (AIEX-725), the unique Canonical Six
pair that is bilateral in both the Cayley-Dickson and Clifford frameworks.

- Verified under this repo's Baez convention before adoption: annihilates in
  BOTH orders (exactly 0.0), collapse identity worst-case error 4e-16.
- Full validation chain re-run green after the change: sedenion_kernel (6/6),
  test_zda_numpy M1–M6 (mirror updated to test Pattern 2 + bilateral check),
  zda_layers T1–T7 (T4 β=0 ≡ baseline diff exactly 0). zda_grid.py Section 1
  updated to match.
- Formal verification: Pattern 2 formally proven as bilateral in CD4, CD5, CD6
  per canonical_six_bilateral_zero_divisors_cd4_cd5_cd6.lean (Lean 4,
  native_decide, zero sorry stubs). This grounds the choice in machine-checked
  proof, not merely computational verification.
- Spec §2.3 requires only "a verified annihilating pair"; the specific pair
  was not pre-registered, and Phase 3 had not launched, so this is a
  pre-launch parameter choice, not a post-hoc change. Phase 2 smoke numbers
  above were produced with the OLD pair (immaterial there: β stayed ≈ 0).
- **2026-07-21 addendum:** `ZD_PAIR` is also, independently, row 52 of
  Reggiani's (arXiv:2411.18881) published table of 84 standard sedenion
  zero divisors, and (per the Canonical Six paper's v1.3 Addendum B,
  `e8_weyl_orbit_unification.lean`) one of five vectors shown to lie in a
  single E₈ Weyl orbit. Three independent lines converging on the same
  pair — full account in `LIE_ALGEBRA_CONNECTIONS.md`.
