"""phase4_X_variety_characterization.py -- does shuffled_structure_tensor
(variant X) have an accessible zero-divisor variety at all?

Motivated by the 2026-07-23 finding (this session, both repo-side and
independently on the chat side) that shuffled_structure_tensor draws 15
INDEPENDENT permutations (one per k), not one consistent relabeling --
so X is an arbitrary bilinear map, not an isomorphic copy of the
sedenions in a misaligned basis. If X has no near-zero-divisor region
reachable at unit norm, then "X never reaches r^2 < 1e-2" (session-close
S3(e)) is not evidence that training failed to find a manifold -- there
is no manifold to find, and the S-vs-X comparison would need reframing
("coherent algebra beats a degenerate/rank-deficient map", not
"coherent algebra beats absence of the phenomenon").

Method: minimize ||x (x) y||^2 over unit-norm x,y in R^16 (torch
autodiff, Adam, many random restarts), separately for the true tensor
(as a positive-control sanity check -- known exact zero divisors exist,
e.g. normalized ZD_PAIR) and each seed's shuffled tensor. Report the
achieved infimum (best over restarts) for each.
"""
import sys
sys.path.insert(0, r"C:\dev\projects\apm-agi_tests")
import numpy as np
import torch

from sedenion_kernel import structure_tensor, basis
from phase4_layers import shuffled_structure_tensor

torch.manual_seed(0)


def mult_norm_sq(T, x, y):
    # T: (16,16,16) torch tensor; x,y: (16,) -> scalar ||x (x) y||^2
    prod = torch.einsum("mij,i,j->m", T, x, y)
    return (prod ** 2).sum()


def minimize_norm(T_np, n_restarts=300, steps=300, lr=0.05):
    T = torch.tensor(T_np, dtype=torch.float64)
    best = float("inf")
    best_xy = None
    for r in range(n_restarts):
        x = torch.randn(16, dtype=torch.float64, requires_grad=True)
        y = torch.randn(16, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([x, y], lr=lr)
        for step in range(steps):
            opt.zero_grad()
            xn = x / x.norm()
            yn = y / y.norm()
            loss = mult_norm_sq(T, xn, yn)
            loss.backward()
            opt.step()
        with torch.no_grad():
            xn = x / x.norm()
            yn = y / y.norm()
            final = mult_norm_sq(T, xn, yn).item()
        if final < best:
            best = final
            best_xy = (xn.detach().clone(), yn.detach().clone())
    return best, best_xy


# --- Positive control: true tensor, known exact zero divisors exist ---
T_true = structure_tensor(16)
# sanity: the known ZD pair should give ~0 directly, no optimization needed
P = basis(3) + basis(12)
Q = basis(5) + basis(10)
Pn, Qn = P / np.linalg.norm(P), Q / np.linalg.norm(Q)
known_zd = np.einsum("mij,i,j->m", T_true, Pn, Qn)
print(f"[sanity] known ZD_PAIR (normalized) ||x(x)y||^2 = "
      f"{(known_zd**2).sum():.3e}  (expect ~0)")

import time
t0 = time.time()
best_true, _ = minimize_norm(T_true, n_restarts=20, steps=150)
print(f"\nTrue T16: best ||x(x)y||^2 over 20 restarts = {best_true:.3e}"
      f"  ({time.time()-t0:.1f}s)", flush=True)

print()
for seed in (1337, 1338, 1339):
    t0 = time.time()
    T_x = shuffled_structure_tensor(seed)
    best_x, _ = minimize_norm(T_x, n_restarts=20, steps=150)
    print(f"Shuffled X seed={seed}: best ||x(x)y||^2 over 20 restarts "
          f"= {best_x:.3e}  ({time.time()-t0:.1f}s)", flush=True)

print()
print("Reading: if X's infimum is ~0 (comparable to True T16's), X has")
print("its own accessible near-zero-divisor region and S3(e)'s finding")
print("(X never reaches r^2<1e-2 during training) is a real claim about")
print("what training found, not an artifact of X having no such region")
print("to find. If X's infimum stays bounded well away from 0, S3(e)")
print("needs reframing.")
