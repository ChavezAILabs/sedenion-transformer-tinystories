"""test_phase4_numpy.py — NumPy mirror of the Phase 4 K3 layer math
(`phase4_layers.py`). Chain link: sedenion_kernel (ground truth) -> this
file (torch-free re-derivation) -> phase4_layers.py (torch port).

Mirrors exactly what the torch layer does per head:
  q,k = x @ Wq, x @ Wk            (d_model -> 16 sedenion coords)
  positional: u(pos) = cos(w*pos) u + sin(w*pos) (L8 u)   [R_8(w*pos)]
  r2_ij = |q_i (x) k_j|^2 / (|q_i|^2 |k_j|^2)
  score = -gamma * r2 ; causal mask ; softmax ; @ v

Checks (Phase 4 replacement anchors for the dead T4, spec SS7):
  [P4M1] tensor-contraction product == cd_mult (ground-truth agreement)
  [P4M2] structural nulls: all 336 two-blade pairs give r = 0 exactly
  [P4M3] K1 guard: separable probe score -> identical attention rows;
         the K3 score -> query-dependent rows
  [P4M4] octonion degeneracy: T16 -> T8 swap makes attention rows
         identical (r == 1 identically)
  [P4M5] R_8 positional: global position shift leaves attention
         invariant (per-head frequency); nulls stay null at all offsets
  [P4M6] causality with the K3 score
  [P4M7] smoothness at the manifold: r2 ~ eps^2, FD gradient converges
"""
import numpy as np
from sedenion_kernel import cd_mult, basis, structure_tensor

rng = np.random.default_rng(3)
T16 = structure_tensor(16)
T8 = structure_tensor(8)
L8 = T16[:, 8, :]
P = basis(3) + basis(12)
Q = basis(5) + basis(10)

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def rot(u, pos, w):
    """R_8(w*pos) applied to rows of u (T,16)."""
    th = w * pos
    return (np.cos(th)[:, None] * u
            + np.sin(th)[:, None] * (u @ L8.T))

def r2_matrix(q, k, T):
    prod = np.einsum("mij,ti,sj->tsm", T, q, k)
    num = (prod ** 2).sum(-1)
    den = (q ** 2).sum(-1)[:, None] * (k ** 2).sum(-1)[None, :]
    return num / den

def k3_attention(q, k, v, gamma, causal=True, T=T16):
    s = -gamma * r2_matrix(q, k, T)
    Tn = s.shape[0]
    if causal:
        s = np.where(np.triu(np.ones((Tn, Tn), bool), 1), -np.inf, s)
    return softmax(s) @ v, softmax(np.where(
        np.triu(np.ones((Tn, Tn), bool), 1), -np.inf, s) if causal else s)

# ---------------------------------------------------------------- [P4M1]
err = 0.0
for _ in range(50):
    x, y = rng.standard_normal(16), rng.standard_normal(16)
    err = max(err, np.abs(np.einsum("mij,i,j->m", T16, x, y)
                          - cd_mult(x, y)).max())
print(f"[P4M1] tensor product == cd_mult:            err {err:.2e}")
assert err < 1e-12

# ---------------------------------------------------------------- [P4M2]
blades = []
for i in range(1, 16):
    for j in range(i + 1, 16):
        for s in (+1.0, -1.0):
            v = np.zeros(16); v[i], v[j] = 1.0, s
            blades.append(v)
B = np.array(blades)
prods = np.einsum("mij,ai,bj->abm", T16, B, B)
null_mask = np.abs(prods).max(axis=2) == 0.0
n = int(null_mask.sum())
assert n == 336, f"expected 336 exact nulls, found {n}"
qi, kj = np.where(null_mask)
r2n = r2_matrix(B[qi[:20]], B[kj[:20]], T16)
worst = max(r2n[t, t] for t in range(20))
print(f"[P4M2] 336 structural nulls, r2 == 0 exactly (worst {worst:.1e})")
assert worst == 0.0

# ---------------------------------------------------------------- [P4M3]
Tn, d = 12, 48
x = rng.standard_normal((Tn, d))
Wq = rng.standard_normal((d, 16)) / np.sqrt(d)
Wk = rng.standard_normal((d, 16)) / np.sqrt(d)
Wv = rng.standard_normal((d, 16)) / np.sqrt(d)
q, k, v = x @ Wq, x @ Wk, x @ Wv
p_hat = P / np.linalg.norm(P); q_hat = Q / np.linalg.norm(Q)
s_k1 = (q @ p_hat)[:, None] + (k @ q_hat)[None, :]      # separable probe
rows_k1 = softmax(s_k1)
spread_k1 = np.abs(rows_k1 - rows_k1[0]).max()
rows_k3 = softmax(-1.0 * r2_matrix(q, k, T16))
spread_k3 = np.abs(rows_k3 - rows_k3[0]).max()
print(f"[P4M3] K1 probe rows identical ({spread_k1:.1e}); "
      f"K3 rows query-dependent ({spread_k3:.3f})")
assert spread_k1 < 1e-12 and spread_k3 > 1e-3

# ---------------------------------------------------------------- [P4M4]
q8, k8 = rng.standard_normal((Tn, 8)), rng.standard_normal((Tn, 8))
r2_oct = r2_matrix(q8, k8, T8)
dev = np.abs(r2_oct - 1.0).max()
rows8 = softmax(-1.0 * r2_oct)
spread8 = np.abs(rows8 - rows8[0]).max()
print(f"[P4M4] octonion swap: |r2-1| {dev:.1e}, row spread {spread8:.1e}")
assert dev < 1e-9 and spread8 < 1e-12

# ---------------------------------------------------------------- [P4M5]
w_h = 0.37                                    # one per-head frequency
pos = np.arange(Tn, dtype=float)
for shift in (5.0, 23.0):
    _, A0 = k3_attention(rot(q, pos, w_h), rot(k, pos, w_h), v, 1.0)
    _, A1 = k3_attention(rot(q, pos + shift, w_h),
                         rot(k, pos + shift, w_h), v, 1.0)
    assert np.abs(A0 - A1).max() < 1e-9
xa, ya = B[qi[0]], B[kj[0]]                   # an exact null pair
worst = max(r2_matrix(rot(xa[None], np.array([m]), w_h),
                      rot(ya[None], np.array([0.0]), w_h), T16)[0, 0]
            for m in np.linspace(0, 40, 17))
print(f"[P4M5] position shift invariance OK; nulls at all offsets "
      f"(worst r2 {worst:.1e})")
assert worst < 1e-24

# ---------------------------------------------------------------- [P4M6]
x2 = x.copy(); x2[7:] += 1.0
o1, _ = k3_attention(x @ Wq, x @ Wk, x @ Wv, 0.9)
o2, _ = k3_attention(x2 @ Wq, x2 @ Wk, x2 @ Wv, 0.9)
dcaus = np.abs(o1[:7] - o2[:7]).max()
print(f"[P4M6] causal mask holds under K3 score:      diff {dcaus:.1e}")
assert dcaus < 1e-12

# ---------------------------------------------------------------- [P4M7]
a0, b0 = 0.9, 0.6
qv = a0 * P + b0 * Q
def r2_eps(e):
    kv = b0 * P + (-a0 + e) * Q
    return r2_matrix(qv[None], kv[None], T16)[0, 0]
r2s = [r2_eps(e) for e in (1e-2, 1e-3, 1e-4)]
ratios = [r2s[0] / r2s[1], r2s[1] / r2s[2]]
g1 = (r2_eps(1e-3 + 1e-5) - r2_eps(1e-3 - 1e-5)) / 2e-5
g2 = (r2_eps(1e-3 + 5e-6) - r2_eps(1e-3 - 5e-6)) / 1e-5
print(f"[P4M7] r2 ~ eps^2 (ratios {ratios[0]:.1f}, {ratios[1]:.1f}); "
      f"FD grad stable ({g1:.4e} vs {g2:.4e})")
assert all(abs(x - 100) < 2 for x in ratios)
assert abs(g1 - g2) / abs(g1) < 1e-4

print("\nAll Phase 4 NumPy mirror checks passed - K3 layer math verified.")
