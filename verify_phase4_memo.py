"""verify_phase4_memo.py — repo-side reproduction of every VERIFIED claim
in PHASE4_kernel_memo.md, against this repo's sedenion_kernel.py.

The memo (Claude Desktop, 2026-07-19) used an independent Cayley-Dickson
implementation; its SS7 requires reproduction against the repo tensor
before anything enters the Phase 4 spec. This script is that gate.
First reproduced green 2026-07-19; numbering follows the memo's SS7 log.

Run:  zda-env\\Scripts\\python.exe verify_phase4_memo.py
"""
import sys
import numpy as np

from sedenion_kernel import cd_mult, cd_conj, basis, structure_tensor

rng = np.random.default_rng(2026)
T16 = structure_tensor(16)          # (x*y)_k = sum_ij T[k,i,j] x_i y_j
T8  = structure_tensor(8)

def mult16(x, y): return np.einsum("kij,i,j->k", T16, x, y)

def softmax_rows(S):
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)

def row_spread(S):
    """max over rows of ||attn_row_i - attn_row_0||_inf after softmax."""
    A = softmax_rows(S)
    return np.abs(A - A[0]).max()

def dc_residual_rms(S):
    """RMS of double-centered matrix (0 iff additively separable)."""
    R = S - S.mean(1, keepdims=True) - S.mean(0, keepdims=True) + S.mean()
    return float(np.sqrt((R ** 2).mean()))

ok = True
def check(name, cond, detail=""):
    global ok
    status = "OK " if cond else "FAIL"
    if not cond: ok = False
    print(f"[{name:<28}] {status} {detail}")

P = basis(3) + basis(12)
Q = basis(5) + basis(10)

# --- 1/2. Pattern 2 bilateral + collapse identity ----------------------
check("1 Pattern2 bilateral", max(np.abs(cd_mult(P, Q)).max(),
                                  np.abs(cd_mult(Q, P)).max()) == 0.0,
      "PQ and QP exactly 0")
worst = 0.0
for _ in range(200):
    a, b, c = rng.standard_normal(3)
    prod = cd_mult(a * P + b * Q, b * P + c * Q)
    target = -2 * b * (a + c) * basis(0)
    worst = max(worst, np.abs(prod - target).max())
check("2 collapse identity x200", worst < 1e-10, f"worst {worst:.1e}")

# --- 3. K1 linear collapse kernel is query-independent ------------------
nq = nk = 32
a_i = rng.standard_normal(nq); c_j = rng.standard_normal(nk); b0 = 0.7
S_k1 = -2 * b0 * (a_i[:, None] + c_j[None, :])
sp = row_spread(S_k1)
check("3 K1 degenerate", sp < 1e-12, f"attn row spread {sp:.1e}")

# --- 4. K2 modulus kernel is NOT query-independent -----------------------
S_k2 = -np.abs(a_i[:, None] + c_j[None, :])
sp = row_spread(S_k2)
check("4 K2 non-degenerate", sp > 1e-3, f"attn row spread {sp:.3f}")

# --- 5. K3 sedenion: interaction + r range -------------------------------
qs = rng.standard_normal((64, 16)); ks = rng.standard_normal((64, 16))
prods = np.einsum("kij,ai,bj->abk", T16, qs, ks)
r = np.linalg.norm(prods, axis=2) / (
    np.linalg.norm(qs, axis=1)[:, None] * np.linalg.norm(ks, axis=1)[None, :])
res = dc_residual_rms(np.log(r))
check("5 K3 sed. interaction", 0.05 < res < 0.3,
      f"dc-residual {res:.3f} (memo 0.139); r range [{r.min():.2f},{r.max():.2f}]")
check("5b norm inflation r>1", r.max() > 1.0, f"max r {r.max():.3f}")
sep = dc_residual_rms(a_i[:, None] + c_j[None, :])
check("5c separable ctrl -> 0", sep < 1e-12, f"dc-residual {sep:.1e}")

# --- 6. Octonions: multiplicative norm, K3 exactly degenerate -------------
worst = 0.0
for _ in range(500):
    x, y = rng.standard_normal(8), rng.standard_normal(8)
    worst = max(worst, abs(np.linalg.norm(cd_mult(x, y))
                           - np.linalg.norm(x) * np.linalg.norm(y)))
check("6 octonion |xy|=|x||y|", worst < 1e-9, f"worst gap {worst:.1e}")
q8 = rng.standard_normal((32, 8)); k8 = rng.standard_normal((32, 8))
p8 = np.einsum("kij,ai,bj->abk", T8, q8, k8)
r8 = np.linalg.norm(p8, axis=2) / (
    np.linalg.norm(q8, axis=1)[:, None] * np.linalg.norm(k8, axis=1)[None, :])
res8 = dc_residual_rms(np.log(r8))
check("6b octonion log r separable", res8 < 1e-12, f"dc-residual {res8:.1e}")
sp8 = row_spread(-3.0 * r8 ** 2)     # end-to-end: g(r^2) attention rows
check("6c octonion K3 attn degen.", sp8 < 1e-9, f"attn row spread {sp8:.1e}")

# --- 7. Structural nulls: 336-pair table + exact null + eps^2 approach ---
pairs = []
for i in range(1, 16):
    for j in range(i + 1, 16):
        for s in (+1.0, -1.0):
            x = basis(i) + s * basis(j)
            for k in range(1, 16):
                for l in range(k + 1, 16):
                    for t in (+1.0, -1.0):
                        y = basis(k) + t * basis(l)
                        if np.abs(mult16(x, y)).max() < 1e-12:
                            pairs.append((x, y))
check("7 exhaustive pair count", len(pairs) == 336, f"found {len(pairs)}")
worst = max(np.linalg.norm(mult16(x, y)) for x, y in pairs)
check("7b |q(x)k|=0 on all 336", worst == 0.0, f"worst |prod| {worst:.1e}")
a0, b1 = 0.9, 0.6
qv = a0 * P + b1 * Q
r2 = []
for eps in (1e-2, 1e-3, 1e-4):
    kv = b1 * P + (-a0 + eps) * Q
    r2.append((np.linalg.norm(cd_mult(qv, kv))
               / (np.linalg.norm(qv) * np.linalg.norm(kv))) ** 2)
ratios = [r2[0] / r2[1], r2[1] / r2[2]]
check("7c r^2 ~ eps^2 at manifold",
      all(abs(x - 100) < 2 for x in ratios),
      f"r2 {r2[0]:.2e}/{r2[1]:.2e}/{r2[2]:.2e}, decade ratios "
      f"{ratios[0]:.1f}, {ratios[1]:.1f}")

# --- 8. Real-part scores: conjugated = Euclidean; plain = signature form --
worst = 0.0
for _ in range(200):
    x, y = rng.standard_normal(16), rng.standard_normal(16)
    worst = max(worst, abs(cd_mult(x, cd_conj(y))[0] - x @ y))
check("8 Re(x (x) conj(y)) = <x,y>", worst < 1e-10, f"worst {worst:.1e}")
sig = np.diag([1.0] + [-1.0] * 15)
check("8b M(e0) = diag(1,-1..-1)", np.array_equal(T16[0], sig), "exact")

# --- 9. K4 family rank = 16 ------------------------------------------------
rank = np.linalg.matrix_rank(T16.reshape(16, 256))
check("9 rank span{M(w)} = 16", rank == 16, f"rank {rank}")

# --- 10. Shuffled-T subspace vs true subspace (repo V4 recipe) -------------
# V4: L_0 = I, L_k (k>=1) random signed permutations. T_shuf[m,k,j]=L[k][m,j].
def v4_tensor(seed):
    g = np.random.default_rng(seed)
    L = np.zeros((16, 16, 16))
    L[0] = np.eye(16)
    for k in range(1, 16):
        perm = g.permutation(16)
        signs = g.integers(0, 2, 16) * 2 - 1
        L[k, np.arange(16), perm] = signs
    return np.transpose(L, (1, 0, 2))       # -> T[m, k, j]

def max_principal_cosine(Ta, Tb):
    A = np.linalg.qr(Ta.reshape(16, -1).T)[0]
    B = np.linalg.qr(Tb.reshape(16, -1).T)[0]
    return float(np.linalg.svd(A.T @ B, compute_uv=False).max())

cosines = [max_principal_cosine(T16, v4_tensor(s)) for s in range(5)]
cos_no0 = [max_principal_cosine(T16[:, 1:, :], v4_tensor(s)[:, 1:, :])
           for s in range(5)]
print(f"[10 shuffled-T overlap        ] INFO 5 draws, max principal cosine "
      f"full {['%.2f' % c for c in cosines]}, excl. e0-slice "
      f"{['%.2f' % c for c in cos_no0]} (memo: 0.44, one draw, own impl)")

# --- 11. Basis left-multiplications are orthogonal --------------------------
worst = max(np.abs(T16[:, i, :].T @ T16[:, i, :] - np.eye(16)).max()
            for i in range(16))
check("11 all 16 L_i orthogonal", worst == 0.0, f"worst dev {worst:.1e}")

print("\n" + ("ALL MEMO CLAIMS REPRODUCED against sedenion_kernel.py"
              if ok else "SOME CHECKS FAILED - see above"))
sys.exit(0 if ok else 1)
