# Sedenion Structure-Tensor Transformer on TinyStories

A pre-registered test of whether embedding sedenion (16-dimensional
hypercomplex) zero-divisor structure into a transformer's attention improves
language modeling. It doesn't — and the negative result is decisive,
verified, and mechanistically interesting.

## The question

Does an attention structure tensor derived from sedenion zero-divisors help a
transformer model language, against a flops-matched dense baseline and a
shuffled-tensor control that removes only the algebraic structure while
holding every other dimension fixed?

## The answer

No, decisively — but not for a boring reason.

- **The primary hypothesis (H4a) fails by 21×.** The structured variant (S)
  trails the flops-matched dense baseline by 0.1600 in validation loss
  against a pre-registered margin of 0.0076.
- **The mechanism engages anyway.** Six attention heads in S reliably
  descend toward the zero-divisor structure across all 3 seeds (5 of 6 in
  layer 0) — it does the thing it was built to do.
- **Doing that thing makes the model worse.** Every dense baseline beats
  both tensor variants on validation loss.
- **The controlled comparison is clean.** S vs. its own shuffled-tensor
  control X — identical in every other dimension (parameter count to the
  digit, flops/token, wall-clock within 6 seconds) — shows a complete
  3/3-seed double dissociation on all three measured axes. This is the one
  comparison in the grid where a single variable differs.
- **Cost exceeds the pre-registered bound.** S runs at 2.39× the primary
  baseline's wall-clock against a 2× ceiling, at identical nominal
  flops/token to a same-flops dense variant.

**Net: the architecture does the thing it was built to do, and doing it
makes the language model worse.**

## Why this is here

Pre-registered hypotheses that fail decisively, with a controlled
dissociation and independent verification, are rare to see published. Every
number in the writeup was independently reproduced from the raw logs (see
`RESULTS_phase4.md` §12) rather than accepted on the original draft's word.
If you're considering a similar structure-tensor approach, this saves the
~18 GPU-hours it took to find out it doesn't help.

## Reproducing

- Seeds fixed at 1337/1338/1339, TF32 disabled (spec §9.2); full config in
  `RESULTS_phase4.md` §2.
- `phase4_grid.py` — training/eval driver; provenance hash recorded in
  `RESULTS_phase4.md`'s header.
- `sedenion_kernel.py`, `phase4_layers.py`, `phase4_train.py` — the
  structure-tensor construction and training loop.
- Verification scripts: `phase4_init_r2_check.py`,
  `phase4_v3b_rigidity_check.py`, `phase4_v3b_cond_regression.py`, and the
  `_null` / `_trained` / `_decision` / `_positional_check` variants.
- Per-run eval logs: `p4_artifacts/*/eval_log.jsonl`,
  `p4_artifacts/*/summary.json`.
- Trained checkpoints are not included; they're regenerable from the fixed
  seeds above (~18 GPU-hours for the full grid). See `.gitignore`.

## Full writeup

`RESULTS_phase4.md` has the complete results, methodology, and four
independent verification passes. Earlier phases (`RESULTS_phase1.md`
through `RESULTS_phase3.md`) and design docs are also in this repo.
