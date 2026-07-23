"""phase4_shuffledT_nulls.py — closes ZDA_phase4_spec.md SS9 item 1:
characterize the accidental null structure of the shuffled structure
tensor (variant X) for the pinned draws, before any run exists.

Recipe pinned here (numpy port of zda_layers.shuffled_left_mult_matrices):
L_0 = I; L_k (k>=1) = independent random signed permutation from
np.random.default_rng(seed). Draw seeds = run seeds (1337, 1338, 1339).

Per draw, against the true-T16 reference row:
  a. two-blade null pairs: count of (e_i + s e_j) (x) (e_k + t e_l) = 0,
     i<j, k<l in 1..15, s,t in {+-1}  (true tensor: 336);
  b. left-factor null directions: count of two-blades x with A_x singular,
     where A_x[m,j] = sum_i T[m,i,j] x_i (any y in null(A_x) annihilates);
  c. reachability: min / median sigma_min(A_x) over 5000 random unit x;
  d. K3 health: r distribution (min/median/max) and double-centered
     log-r residual on 64x64 random pairs (0 would mean degenerate),
     plus max principal cosine vs the true tensor subspace.
"""
import numpy as np

from sedenion_kernel import structure_tensor

T16 = structure_tensor(16)

def shuffled_tensor(seed):
    g = np.random.default_rng(seed)
    L = np.zeros((16, 16, 16))
    L[0] = np.eye(16)
    for k in range(1, 16):
        perm = g.permutation(16)
        signs = g.integers(0, 2, 16) * 2 - 1
        L[k, np.arange(16), perm] = signs
    return np.transpose(L, (1, 0, 2))          # T[m, i, j], i = left factor

def two_blades():
    V = []
    for i in range(1, 16):
        for j in range(i + 1, 16):
            for s in (+1.0, -1.0):
                v = np.zeros(16); v[i] = 1.0; v[j] = s
                V.append(v)
    return np.array(V)                          # (210, 16)

B = two_blades()

def characterize(T, label, rng):
    # a. exhaustive two-blade pair nulls
    prods = np.einsum("mij,ai,bj->abm", T, B, B)
    n_pairs = int((np.abs(prods).max(axis=2) < 1e-12).sum())
    # b. two-blade left-null directions
    A = np.einsum("mij,ai->amj", T, B)          # (210, 16, 16)
    sv = np.linalg.svd(A, compute_uv=False)     # (210, 16)
    n_leftnull = int((sv[:, -1] < 1e-12).sum())
    # c. sigma_min over random unit x
    xs = rng.standard_normal((5000, 16))
    xs /= np.linalg.norm(xs, axis=1, keepdims=True)
    Ax = np.einsum("mij,ai->amj", T, xs)
    smin = np.linalg.svd(Ax, compute_uv=False)[:, -1]
    # d. K3 health on random pairs
    qs = rng.standard_normal((64, 16)); ks = rng.standard_normal((64, 16))
    p = np.einsum("mij,ai,bj->abm", T, qs, ks)
    r = np.linalg.norm(p, axis=2) / (np.linalg.norm(qs, axis=1)[:, None]
                                     * np.linalg.norm(ks, axis=1)[None, :])
    lg = np.log(r)
    dc = lg - lg.mean(1, keepdims=True) - lg.mean(0, keepdims=True) + lg.mean()
    dc_rms = float(np.sqrt((dc ** 2).mean()))
    # principal-angle overlap with true tensor
    Aq = np.linalg.qr(T.reshape(16, -1).T)[0]
    Bq = np.linalg.qr(T16.reshape(16, -1).T)[0]
    cos = float(np.linalg.svd(Aq.T @ Bq, compute_uv=False).max())
    print(f"{label:<12} {n_pairs:>7} {n_leftnull:>9} "
          f"{smin.min():>10.2e} {np.median(smin):>10.3f} "
          f"[{r.min():.3f},{np.median(r):.3f},{r.max():.3f}]"
          f"{dc_rms:>8.3f} {cos:>6.2f}")
    return n_pairs, n_leftnull, dc_rms

print(f"{'draw':<12} {'2bl-pairs':>7} {'2bl-xnull':>9} "
      f"{'min s_min':>10} {'med s_min':>10} {'r [min,med,max]':>21}"
      f"{'dcRMS':>8} {'cosT':>6}")
rng = np.random.default_rng(99)
ref = characterize(T16, "TRUE T16", rng)
assert ref[0] == 336, "true-tensor reference must reproduce 336 pairs"
results = {}
for seed in (1337, 1338, 1339):
    results[seed] = characterize(shuffled_tensor(seed), f"X seed {seed}", rng)

print()
for seed, (np_, nx, dc) in results.items():
    verdict = []
    verdict.append("has exact two-blade nulls" if np_ > 0 else
                   "NO two-blade nulls (X lacks the structural-null property"
                   " at two-blade level)")
    verdict.append("non-degenerate interaction" if dc > 0.01 else
                   "DEGENERATE (near-separable) — reject draw")
    print(f"X seed {seed}: {np_} two-blade null pairs "
          f"({np_/336:.2f}x true), {nx} null left-directions; "
          + "; ".join(verdict))
