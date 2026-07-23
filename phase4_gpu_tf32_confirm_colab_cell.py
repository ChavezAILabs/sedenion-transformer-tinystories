# ZDA Phase 4 -- "one-cell GPU confirm" (ZDA_phase4_spec.md SS9 item 2).
# Paste this whole cell into Colab (GPU runtime: Runtime > Change runtime
# type > GPU) and run it first, before any training. No repo checkout
# needed -- self-contained, derived from phase4_gpu_tf32_confirm.py in
# the apm-agi_tests repo (that file is the canonical/rerunnable version;
# this is a paste-ready copy with sedenion_kernel.py's two functions
# inlined so it needs nothing but numpy+torch).
#
# What it checks: whether TF32 (left on by PyTorch's CUDA default,
# confirmed absent as an explicit setting anywhere in this repo until
# this session) silently reproduces the same near-manifold gradient-
# killing floor that fp16 storage was already shown to cause on CPU
# (2026-07-19: fp16 floors at eps~1e-3, bf16 at eps~1e-2, fp32 faithful
# to eps=1e-5). phase4_train.py now sets allow_tf32=False on CUDA; this
# cell is what actually verifies that matters on real hardware, rather
# than assuming it from the shared 10-bit-mantissa argument.

import numpy as np
import torch

N_ALG = 16
EPS_GRID = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
N_PAIRS_PER_EPS = 512

# --- inlined from sedenion_kernel.py (Baez convention) ---
def cd_conj(x):
    out = -x.copy(); out[0] = x[0]; return out

def cd_mult(x, y):
    n = len(x)
    if n == 1:
        return x * y
    h = n // 2
    a, b = x[:h], x[h:]
    c, d = y[:h], y[h:]
    real = cd_mult(a, c) - cd_mult(cd_conj(d), b)
    imag = cd_mult(d, a) + cd_mult(b, cd_conj(c))
    return np.concatenate([real, imag])

def basis(i, dim=16):
    e = np.zeros(dim); e[i] = 1.0; return e

def structure_tensor(dim=16):
    T = np.zeros((dim, dim, dim))
    for i in range(dim):
        for j in range(dim):
            T[:, i, j] = cd_mult(basis(i, dim), basis(j, dim))
    return T
# --- end inlined section ---

def zd_pair():
    P = np.zeros(N_ALG); P[3] = 1.0; P[12] = 1.0     # e3+e12
    Q = np.zeros(N_ALG); Q[5] = 1.0; Q[10] = 1.0      # e5+e10
    assert np.abs(cd_mult(P, Q)).max() < 1e-12, "ZD_PAIR must annihilate exactly"
    return P, Q

def build_sweep(rng):
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
    return np.stack(qs), np.stack(ks)

def r2_via_model_contraction(T, q, k, requires_grad=False):
    if requires_grad:
        q = q.clone().requires_grad_(True)
        k = k.clone().requires_grad_(True)
    k_rot = torch.einsum("mij,esj->esim", T, k)
    prod = torch.einsum("eti,etim->etm", q, k_rot)
    num = prod.pow(2).sum(-1)
    den = q.pow(2).sum(-1) * k.pow(2).sum(-1) + 1e-12
    r2 = num / den
    if requires_grad:
        return r2, q, k
    return r2

def find_floor(eps_grid, r2_median):
    floor_eps = None
    for i in range(len(eps_grid) - 1, 0, -1):
        if r2_median[i] == 0.0:
            floor_eps = eps_grid[i]; continue
        slope = (np.log10(r2_median[i]) - np.log10(r2_median[i - 1])) / \
                (np.log10(eps_grid[i]) - np.log10(eps_grid[i - 1]))
        if slope < 1.5:
            floor_eps = eps_grid[i]
        else:
            break
    return floor_eps

def run_condition(label, T_np, q_np, k_np, dtype, device, tf32_matmul=None, check_grad=False):
    if device == "cuda" and tf32_matmul is not None:
        torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
        torch.backends.cudnn.allow_tf32 = tf32_matmul
    T = torch.tensor(T_np, dtype=dtype, device=device)
    q = torch.tensor(q_np, dtype=dtype, device=device)
    k = torch.tensor(k_np, dtype=dtype, device=device)
    if check_grad:
        r2, qg, kg = r2_via_model_contraction(T, q, k, requires_grad=True)
        loss = r2[-1].sum()
        loss.backward()
        grad_finite = torch.isfinite(qg.grad).all().item()
        grad_norm = qg.grad.norm().item()
    else:
        r2 = r2_via_model_contraction(T, q, k)
        grad_finite, grad_norm = None, None
    r2_np = r2.detach().float().cpu().numpy()
    r2_median = np.median(r2_np, axis=1)
    floor = find_floor(EPS_GRID, r2_median)
    print(f"  [{label:<28}] r2 median per eps: " + " ".join(f"{v:.3e}" for v in r2_median)
          + f"   floor={'none (faithful to ' + str(EPS_GRID[-1]) + ')' if floor is None else floor}")
    if check_grad:
        print(f"      grad@eps={EPS_GRID[-1]}: finite={grad_finite} norm={grad_norm:.3e} "
              f"{'DEAD' if grad_norm < 1e-20 else 'live'}")
    return r2_median, floor

rng = np.random.default_rng(2026)
T16 = structure_tensor(N_ALG)
q_np, k_np = build_sweep(rng)

cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}" + (f"  ({torch.cuda.get_device_name(0)})" if cuda_available else ""))
assert cuda_available, "No GPU runtime -- Runtime > Change runtime type > GPU, then rerun this cell."
print(f"eps grid: {EPS_GRID}\n")

print("Reference (fp64, CPU):")
run_condition("fp64 gold", T16, q_np, k_np, torch.float64, "cpu")

print("\nGPU sweep:")
_, floor_tf32_off = run_condition("fp32, TF32 OFF (current guard)", T16, q_np, k_np, torch.float32, "cuda", tf32_matmul=False, check_grad=True)
_, floor_tf32_on  = run_condition("fp32, TF32 ON (pre-fix default)", T16, q_np, k_np, torch.float32, "cuda", tf32_matmul=True, check_grad=True)
run_condition("fp16 (CUDA)", T16, q_np, k_np, torch.float16, "cuda", tf32_matmul=False, check_grad=True)
run_condition("bf16 (CUDA)", T16, q_np, k_np, torch.bfloat16, "cuda", tf32_matmul=False, check_grad=True)

print("\nVerdict:")
if floor_tf32_off is None:
    print(f"  PASS: fp32/TF32-off is faithful to the full grid (eps={EPS_GRID[-1]}) -- the guard is doing its job.")
else:
    print(f"  FAIL: fp32/TF32-off floors at eps={floor_tf32_off} -- do not run real training until this is understood.")
if floor_tf32_on is not None and floor_tf32_off is None:
    print(f"  CONFIRMS the fix was load-bearing: TF32-on floors at eps={floor_tf32_on} where TF32-off is fully faithful.")
elif floor_tf32_on is None:
    print("  NOTE: TF32-on did not floor at this shape/hardware -- inconclusive on whether TF32 was actually invoked "
          "(cuBLAS heuristics are shape-dependent); the guard is kept regardless, cost is negligible at Phase 4 dims.")
