# RESULTS_phase1.md — ZDA Phase 1 Validation

**Date:** 2026-07-16
**Machine:** Windows 11 Home (10.0.26200), CPU only — `nvidia-smi` not present, no NVIDIA GPU detected, so the CPU torch build was installed per HANDOFF instructions.
**Verdict: ALL THREE FILES PASS. Zero code changes were required — `zda_layers.py` ran clean on its first-ever execution.**

## Environment

- Python 3.13.2 (venv `zda-env`, created via `py -3.13 -m venv zda-env`)
- torch 2.13.0+cpu (`torch.cuda.is_available()` → False)
- numpy 2.5.1
- Full pin list in `requirements.txt` (pip freeze)

## 1. `python sedenion_kernel.py` — verbatim output

```
[1] structure tensor == recursive mult:   max err 8.88e-16
[2] quaternion |ab| = |a||b|:            gap 2.22e-16
[2] octonion   |ab| = |a||b|:            gap 1.78e-15
[3] sedenion norm generally NOT mult.:     gap 7.29e-02 (expected > 0)
[4] simple zero-divisor products found:    336
    example: (e1 + e10) * (e4 - e15) = 0
[5] bilateral collapse identity holds for: 336/336 pairs
[6] ZD-gate primitive verified: |prod|=0 iff a=-c (m0=0.0e+00, m1=4.16)

All checks passed. Structure tensor is safe to port to PyTorch.
```

All 6 checks pass. 336 annihilating products found; collapse identity holds 336/336 — matches the sandbox run exactly.

## 2. `python test_zda_numpy.py` — verbatim output

```
[M1] PHM-16 == matrix of learned sedenions:  err 1.78e-15
     param count = dense/16 confirmed (96 vs 1536)
[M2] ||(aP+bQ)(bP+cQ)|| == 2|b||a+c| for frame pair (200 draws)
[M3] beta=0 gated attention == standard:     err 0.00e+00
[M4] causal mask holds with gate active (beta=0.7)
[M5] dL/dbeta at 0: FD=+2.264983, half-eps=+2.264983 (agree to 4.9e-08; nonzero -> gate trainable)
[M6] RoPE relative property: -9.3159429232 == -9.3159429232

All NumPy mirror checks passed — zda_layers.py math is verified.
```

[M1]–[M6] all pass.

## 3. `python zda_layers.py` — verbatim output (FIRST EXECUTION)

```
[T1] ZD pair annihilates under torch build            OK
[T2] PHMLinear params = dense/16; W == kron sum       OK
[T3] gradcheck PHMLinear                              OK
[T4] V2(beta=0) == baseline (max diff 0.0e+00)      OK
[T5] d(loss)/d(beta) nonzero at init (1.585e+00)     OK
[T6] causal mask holds under gate                     OK
[T7] all variant constructors forward cleanly        OK

All Phase 1 tests passed.
```

Expected results confirmed:
- T1: ZD pair annihilates under the torch structure-tensor build.
- T2: PHM param count = dense/16; W == explicit kron sum.
- T3: float64 gradcheck passes.
- T4: **V2(β=0) == baseline with max diff 0.0 — the load-bearing invariant holds exactly (bit-identical), better than the 1e-6 requirement.**
- T5: ∂loss/∂β = 1.585 at init — gate is trainable.
- T6: causal mask holds with gate active.
- T7: all five variant configurations forward cleanly.

## Fixes made

**None.** No plumbing or math changes; no tolerances touched.

## Notes / surprises

- The β=0 invariant (T4) held at exactly 0.0, not merely within 1e-6 — the gate term is an additive `β·|a+c|` with β a zero tensor, so the logits are bit-identical to baseline. Same for [M3] in the NumPy mirror.
- torch 2.13.0 CPU on Python 3.13 accepted the code without deprecation warnings.
- No GPU on this machine: Phase 2 smoke run will be CPU-only (fine per HANDOFF §5); the full 18-run grid stays on Colab/Kaggle as specified.
