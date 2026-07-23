"""phase4_gpu_tf32_confirm.py -- the "one-cell GPU confirm" from
ZDA_phase4_spec.md SS9 item 2, the last open piece of the fp32-hygiene
rule. Meant to be run as a single Colab cell, first thing after
connecting to a GPU runtime, before any real training.

Background (already established on CPU, 2026-07-19, not rerun here):
fp16 storage of q/k quantizes the score path's near-manifold distance to
a ~1e-3 floor (r^2 reads exactly 0 below it, gradient dead); bf16's floor
is ~1e-2; fp32 is faithful to 1e-5. That CPU sweep never exercised TF32,
because TF32 is a CUDA matmul/tensor-core behavior with no CPU analogue
-- PyTorch's `torch.backends.cuda.matmul.allow_tf32` defaults True on
Ampere+ (confirmed absent as an explicit setting anywhere in this repo
until this session's fix to phase4_train.py:127-135). TF32 has the same
10-bit mantissa as fp16, so the theoretical prediction is that leaving
it on would reproduce fp16's floor even though the tensors are nominally
fp32 -- this script is what actually checks that on real hardware,
rather than assuming it.

What it does: builds a batch of (q, k) pairs near the true ZD_PAIR
(e3+e12, e5+e10), at controlled distances eps in {1e-1, ..., 1e-5} from
the annihilation manifold, shaped like a real attention score batch
(not a single toy vector -- TF32 dispatch is shape/size dependent), and
computes r^2 through the *exact* two-step einsum contraction
K3Attention.scores() uses (same T buffer, same contraction order), under:
  - fp64 (gold reference, CPU)
  - fp32, TF32 explicitly OFF   <- what phase4_train.py now sets on CUDA
  - fp32, TF32 explicitly ON    <- what it would have silently done before
  - fp16
  - bf16
For each, reports the eps at which r^2 stops tracking eps^2 (the "floor"),
plus a gradient-liveness check (d(r^2)/dq finite and nonzero at eps=1e-5).

PASS bar (spec SS9.2, restated concretely): fp32/TF32-off must be
faithful to eps=1e-5 (no floor at or above it, gradients live). If
fp32/TF32-on shows a floor comparable to fp16's (~1e-3), that CONFIRMS
the guard added to phase4_train.py this session was load-bearing, not
defensive boilerplate.

Runs on CPU too (TF32 branches degrade to a no-op / are skipped with a
note) so the sweep logic itself can be smoke-tested before GPU time is
spent -- that smoke test was run locally as part of writing this file;
the CUDA-specific comparison (the actual point of this script) has NOT
been executed anywhere yet and needs real hardware.

Run:  zda-env\\Scripts\\python.exe phase4_gpu_tf32_confirm.py
On Colab (GPU runtime): !python phase4_gpu_tf32_confirm.py
"""
import numpy as np
import torch

from sedenion_kernel import cd_mult, structure_tensor

N_ALG = 16
EPS_GRID = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
N_PAIRS_PER_EPS = 512     # batch size per eps bucket -- realistic attention-score scale


def zd_pair():
    P = np.zeros(N_ALG); P[3] = 1.0; P[12] = 1.0     # e3+e12
    Q = np.zeros(N_ALG); Q[5] = 1.0; Q[10] = 1.0      # e5+e10
    assert np.abs(cd_mult(P, Q)).max() < 1e-12, "ZD_PAIR must annihilate exactly"
    return P, Q


def build_sweep(rng):
    """(n_eps, N_PAIRS_PER_EPS, 16) q,k arrays at each eps, fp64, CPU/numpy.
    q = normalize(P + eps*v), k = normalize(Q + eps*w), v/w random unit
    directions -- generic perturbation, not tangent-restricted, matching
    how the CPU sweep in SS9.2 was described (perturbation *from* the
    manifold, not *along* it)."""
    P, Q = zd_pair()
    qs, ks = [], []
    for eps in EPS_GRID:
        v = rng.standard_normal((N_PAIRS_PER_EPS, N_ALG))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        w = rng.standard_normal((N_PAIRS_PER_EPS, N_ALG))
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        q = P[None, :] + eps * v
        k = Q[None, :] + eps * w
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        k /= np.linalg.norm(k, axis=1, keepdims=True)
        qs.append(q); ks.append(k)
    return np.stack(qs), np.stack(ks)   # (n_eps, N_PAIRS_PER_EPS, 16) each


def r2_via_model_contraction(T, q, k, requires_grad=False):
    """Identical contraction to K3Attention.scores(): two-step einsum
    through the structure tensor, same order, same shapes-up-to-batching.
    q, k: (n_eps, N, 16) torch tensors, T: (16,16,16) torch tensor, all
    same dtype/device. Returns r2: (n_eps, N)."""
    if requires_grad:
        q = q.clone().requires_grad_(True)
        k = k.clone().requires_grad_(True)
    k_rot = torch.einsum("mij,esj->esim", T, k)      # (n_eps, N, 16, 16)
    prod = torch.einsum("eti,etim->etm", q, k_rot)    # (n_eps, N, 16)
    num = prod.pow(2).sum(-1)
    den = q.pow(2).sum(-1) * k.pow(2).sum(-1) + 1e-12
    r2 = num / den
    if requires_grad:
        return r2, q, k
    return r2


def find_floor(eps_grid, r2_median):
    """Smallest eps (from the small end) where r2 stops decreasing
    log-linearly with eps -- i.e. the quantization floor. A clean r2~eps^2
    manifold gives log-log slope ~2 between every consecutive pair; a
    floor shows up as a slope collapsing toward 0 (r2 stops shrinking)
    or an exact-zero read (hard underflow to the dtype's floor). NOTE:
    small absolute r2 is NOT itself evidence of a floor -- at eps=1e-5,
    r2~1e-10 is the *expected*, correct value, caught a real bug in an
    earlier version of this function that flagged exactly that (see
    phase4_gpu_tf32_confirm.py's CPU smoke-test note)."""
    floor_eps = None
    for i in range(len(eps_grid) - 1, 0, -1):
        if r2_median[i] == 0.0:
            floor_eps = eps_grid[i]
            continue
        slope = (np.log10(r2_median[i]) - np.log10(r2_median[i - 1])) / \
                (np.log10(eps_grid[i]) - np.log10(eps_grid[i - 1]))
        if slope < 1.5:   # clean eps^2 gives ~2.0; allow noise headroom
            floor_eps = eps_grid[i]
        else:
            break
    return floor_eps


def run_condition(label, T_np, q_np, k_np, dtype, device,
                   tf32_matmul=None, check_grad=False):
    if device == "cuda" and tf32_matmul is not None:
        torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
        torch.backends.cudnn.allow_tf32 = tf32_matmul
    elif device != "cuda" and tf32_matmul is not None:
        print(f"  [{label}] skipped -- TF32 is CUDA-only, no meaningful "
              f"comparison on {device}")
        return None

    T = torch.tensor(T_np, dtype=dtype, device=device)
    q = torch.tensor(q_np, dtype=dtype, device=device)
    k = torch.tensor(k_np, dtype=dtype, device=device)

    if check_grad:
        r2, qg, kg = r2_via_model_contraction(T, q, k, requires_grad=True)
        loss = r2[-1].sum()   # smallest-eps bucket, where gradient liveness matters most
        loss.backward()
        grad_finite = torch.isfinite(qg.grad).all().item()
        grad_norm = qg.grad.norm().item()
    else:
        r2 = r2_via_model_contraction(T, q, k)
        grad_finite, grad_norm = None, None

    r2_np = r2.detach().float().cpu().numpy()
    r2_median = np.median(r2_np, axis=1)
    floor = find_floor(EPS_GRID, r2_median)
    print(f"  [{label:<28}] r2 median per eps: "
          + " ".join(f"{v:.3e}" for v in r2_median)
          + f"   floor={'none (faithful to ' + str(EPS_GRID[-1]) + ')' if floor is None else floor}")
    if check_grad:
        print(f"      grad@eps={EPS_GRID[-1]}: finite={grad_finite} "
              f"norm={grad_norm:.3e} {'DEAD' if grad_norm < 1e-20 else 'live'}")
    return r2_median, floor


def main():
    rng = np.random.default_rng(2026)
    T16 = structure_tensor(N_ALG)
    q_np, k_np = build_sweep(rng)   # fp64 numpy, (n_eps, N, 16)

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}"
          + (f"  ({torch.cuda.get_device_name(0)})" if cuda_available else ""))
    print(f"eps grid: {EPS_GRID}\n")

    print("Reference (fp64, CPU):")
    ref_median, ref_floor = run_condition("fp64 gold", T16, q_np, k_np,
                                           torch.float64, "cpu")

    print("\nCPU fp32 (no TF32 concept on CPU):")
    run_condition("fp32 (CPU)", T16, q_np, k_np, torch.float32, "cpu")

    if cuda_available:
        print("\nGPU sweep:")
        _, floor_tf32_off = run_condition("fp32, TF32 OFF (current guard)",
                                           T16, q_np, k_np, torch.float32,
                                           "cuda", tf32_matmul=False, check_grad=True)
        _, floor_tf32_on = run_condition("fp32, TF32 ON (pre-fix default)",
                                          T16, q_np, k_np, torch.float32,
                                          "cuda", tf32_matmul=True, check_grad=True)
        run_condition("fp16 (CUDA)", T16, q_np, k_np, torch.float16,
                      "cuda", tf32_matmul=False, check_grad=True)
        run_condition("bf16 (CUDA)", T16, q_np, k_np, torch.bfloat16,
                      "cuda", tf32_matmul=False, check_grad=True)

        print("\nVerdict:")
        if floor_tf32_off is None:
            print("  PASS: fp32/TF32-off is faithful to the full grid "
                  f"(eps={EPS_GRID[-1]}) -- phase4_train.py's guard is doing its job.")
        else:
            print(f"  FAIL: fp32/TF32-off floors at eps={floor_tf32_off} -- "
                  "the guard alone is not sufficient on this hardware/PyTorch "
                  "version; do not run real training until this is understood.")
        if floor_tf32_on is not None and floor_tf32_off is None:
            print(f"  CONFIRMS the fix was load-bearing: TF32-on floors at "
                  f"eps={floor_tf32_on} on the same hardware where TF32-off "
                  "is fully faithful -- leaving the pre-fix default in place "
                  "would have silently broken the score path's gradient "
                  "near the manifold.")
        elif floor_tf32_on is None:
            print("  NOTE: TF32-on did not floor at this batch size/shape on "
                  "this hardware -- inconclusive on whether TF32 was ever "
                  "actually invoked here (cuBLAS heuristics are shape- and "
                  "size-dependent); the guard is kept regardless, since it "
                  "costs negligible time at Phase 4 dims and removes the "
                  "ambiguity entirely rather than relying on heuristics.")
    else:
        print("\nNo CUDA device -- this is the CPU dry-run of the sweep "
              "logic only. The actual point of this script (TF32 on real "
              "hardware) has not been checked. Run on a GPU runtime.")


if __name__ == "__main__":
    main()
