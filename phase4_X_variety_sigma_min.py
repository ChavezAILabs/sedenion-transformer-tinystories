"""phase4_X_variety_sigma_min.py -- N1/N3-adjacent: re-run the chat-side
sigma_min characterization (REPLY_to_ClaudeCode_2026-07-24.md,
FINDINGS_zd_variety_characterization_2026-07-24.md) against the REAL
repo artifacts, not the two guessed "readings" of the shuffle spec used
there (that session had no repo access).

For fixed unit x, y -> x (x) y is linear: (x(x)y)_m = sum_ij T[m,i,j] x_i
y_j = sum_j M_x[m,j] y_j where M_x[m,j] = sum_i T[m,i,j] x_i. So
  min over unit y of ||x (x) y|| = sigma_min(M_x)
and inf over unit x of that is the ZD-variety accessibility question.

Two things computed, both directly against sedenion_kernel.structure_tensor
and phase4_layers.shuffled_structure_tensor (the actual functions used
throughout this project, seeds 1337/1338/1339 -- the real seeds, not a
class-level sample):

  (A) infimum of sigma_min(M_x) over unit x (gradient descent via
      autograd through torch.linalg.svdvals, many restarts) -- repeats
      FINDINGS S3.1/3.2.
  (B) free-sphere accessibility: sigma_min(M_x) at many UNIFORM random
      unit x, report median/p05/frac<thresholds -- repeats FINDINGS S3.3,
      the more decisive statistic per the chat-side reply.
"""
import sys
sys.path.insert(0, r"C:\dev\projects\apm-agi_tests")
import time
import numpy as np
import torch

from sedenion_kernel import structure_tensor, basis
from phase4_layers import shuffled_structure_tensor

torch.manual_seed(0)


def M_x(T, x):
    # T: (16,16,16) [m,i,j]; x: (16,) -> M_x: (16,16) [m,j]
    return torch.einsum("mij,i->mj", T, x)


def sigma_min(T, x):
    return torch.linalg.svdvals(M_x(T, x))[-1]


def infimum_sigma_min(T_np, n_restarts=30, steps=200, lr=0.05):
    T = torch.tensor(T_np, dtype=torch.float64)
    n = T.shape[-1]
    best = float("inf")
    for r in range(n_restarts):
        x = torch.randn(n, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([x], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            xn = x / x.norm()
            loss = sigma_min(T, xn)
            loss.backward()
            opt.step()
        with torch.no_grad():
            xn = x / x.norm()
            val = sigma_min(T, xn).item()
        best = min(best, val)
    return best


def free_sphere_stats(T_np, n_samples=20000):
    T = torch.tensor(T_np, dtype=torch.float64)
    g = torch.Generator().manual_seed(12345)
    X = torch.randn(n_samples, 16, dtype=torch.float64, generator=g)
    X = X / X.norm(dim=1, keepdim=True)
    vals = torch.empty(n_samples, dtype=torch.float64)
    # batched M_x then batched smallest singular value
    Mx_all = torch.einsum("mij,ni->nmj", T, X)   # (n,16,16)
    svals = torch.linalg.svdvals(Mx_all)         # (n,16), descending
    vals = svals[:, -1]
    v = vals.numpy()
    return {
        "median": float(np.median(v)),
        "p05": float(np.percentile(v, 5)),
        "frac<1e-1": float((v < 1e-1).mean()),
        "frac<1e-2": float((v < 1e-2).mean()),
        "frac<1e-3": float((v < 1e-3).mean()),
    }


# --- method validation, per FINDINGS S2 (division algebras -> infimum=1) ---
def struct_tensor_octonion():
    return structure_tensor(8)


def struct_tensor_quaternion():
    return structure_tensor(4)


print("=== Method validation (division algebras, infimum should = 1) ===")
for name, T in [("quaternion", struct_tensor_quaternion()),
                 ("octonion", struct_tensor_octonion())]:
    inf_val = infimum_sigma_min(T, n_restarts=15, steps=150)
    print(f"{name}: infimum sigma_min = {inf_val:.6f}  (expect 1.0)")

# known ZD pair sanity on the true sedenion tensor
T_true = structure_tensor(16)
P = basis(3) + basis(12)
Pn = P / np.linalg.norm(P)
Mp = M_x(torch.tensor(T_true, dtype=torch.float64),
         torch.tensor(Pn, dtype=torch.float64))
sm = torch.linalg.svdvals(Mp)[-1].item()
print(f"\nsigma_min(M_x) at x=normalized ZD_PAIR.P = {sm:.3e}  (expect ~0)")

print("\n=== (A) Infimum of sigma_min over unit x -- REAL repo tensors ===")
t0 = time.time()
inf_true = infimum_sigma_min(T_true, n_restarts=20, steps=200)
print(f"True T16 (sedenion_kernel.structure_tensor(16)): "
      f"infimum = {inf_true:.3e}  ({time.time()-t0:.1f}s)", flush=True)

for seed in (1337, 1338, 1339):
    t0 = time.time()
    T_x = shuffled_structure_tensor(seed)
    inf_x = infimum_sigma_min(T_x, n_restarts=20, steps=200)
    print(f"Shuffled X seed={seed} (phase4_layers.shuffled_structure_tensor): "
          f"infimum = {inf_x:.3e}  ({time.time()-t0:.1f}s)", flush=True)

print("\n=== (B) Free-sphere accessibility -- REAL repo tensors, 20000 samples ===")
stats_true = free_sphere_stats(T_true)
print(f"True T16: {stats_true}", flush=True)
for seed in (1337, 1338, 1339):
    T_x = shuffled_structure_tensor(seed)
    stats_x = free_sphere_stats(T_x)
    ratio = stats_true["frac<1e-1"] and (stats_x["frac<1e-1"] / max(stats_true["frac<1e-1"], 1e-12))
    print(f"Shuffled X seed={seed}: {stats_x}  "
          f"(frac<1e-1 ratio X/True = {ratio:.1f}x)", flush=True)
