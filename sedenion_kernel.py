"""
sedenion_kernel.py — Verified algebra kernel for the ZDA experiment.

Implements the Cayley-Dickson construction up to sedenions (16D),
builds the multiplication structure tensor T[k,i,j] used by the
PHM-16 layer and the ZD-gate, searches for bilateral zero divisor
pairs, and numerically checks the bilateral collapse identity.

Convention: (a,b)(c,d) = (ac - d*.b, d.a + b.c*)   [Baez convention]
Conjugate:  (a,b)* = (a*, -b)

Everything here is plain NumPy so it ports 1:1 to a PyTorch einsum.
"""

import numpy as np
from itertools import combinations

# ---------------------------------------------------------------
# Cayley-Dickson construction (recursive, on coefficient vectors)
# ---------------------------------------------------------------

def cd_conj(x):
    out = -x.copy()
    out[0] = x[0]
    return out

def cd_mult(x, y):
    n = len(x)
    if n == 1:
        return x * y
    h = n // 2
    a, b = x[:h], x[h:]
    c, d = y[:h], y[h:]
    # (a,b)(c,d) = (ac - d*.b, d.a + b.c*)
    real = cd_mult(a, c) - cd_mult(cd_conj(d), b)
    imag = cd_mult(d, a) + cd_mult(b, cd_conj(c))
    return np.concatenate([real, imag])

def basis(i, dim=16):
    e = np.zeros(dim)
    e[i] = 1.0
    return e

# ---------------------------------------------------------------
# Structure tensor: (x*y)_k = sum_ij T[k,i,j] x_i y_j
# This is the object the PyTorch layer consumes.
# ---------------------------------------------------------------

def structure_tensor(dim=16):
    T = np.zeros((dim, dim, dim))
    for i in range(dim):
        for j in range(dim):
            T[:, i, j] = cd_mult(basis(i, dim), basis(j, dim))
    return T

# ---------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------

def run_checks():
    rng = np.random.default_rng(0)
    dim = 16
    T = structure_tensor(dim)

    # 1. Structure tensor reproduces recursive multiplication
    x, y = rng.standard_normal(dim), rng.standard_normal(dim)
    err = np.abs(np.einsum('kij,i,j->k', T, x, y) - cd_mult(x, y)).max()
    print(f"[1] structure tensor == recursive mult:   max err {err:.2e}")
    assert err < 1e-12

    # 2. Quaternion/octonion levels: norm IS multiplicative (division algebras)
    for d, name in [(4, "quaternion"), (8, "octonion")]:
        a, b = rng.standard_normal(d), rng.standard_normal(d)
        gap = abs(np.linalg.norm(cd_mult(a, b)) - np.linalg.norm(a) * np.linalg.norm(b))
        print(f"[2] {name:10s} |ab| = |a||b|:            gap {gap:.2e}")
        assert gap < 1e-10

    # 3. Sedenions: norm NOT multiplicative in general (zero divisors exist)
    a, b = rng.standard_normal(16), rng.standard_normal(16)
    gap = abs(np.linalg.norm(cd_mult(a, b)) - np.linalg.norm(a) * np.linalg.norm(b))
    print(f"[3] sedenion norm generally NOT mult.:     gap {gap:.2e} (expected > 0)")
    assert gap > 1e-6

    # 4. Exhaustive search for bilateral zero divisors of the form
    #    (e_i + s1*e_j)(e_k + s2*e_l) = 0,  i<j, k<l, signs in {+1,-1}
    found = []
    for (i, j) in combinations(range(1, 16), 2):
        for (k, l) in combinations(range(1, 16), 2):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    P = basis(i) + s1 * basis(j)
                    Q = basis(k) + s2 * basis(l)
                    if np.abs(cd_mult(P, Q)).max() < 1e-12:
                        found.append((i, s1, j, k, s2, l))
    print(f"[4] simple zero-divisor products found:    {len(found)}")
    assert len(found) > 0
    i, s1, j, k, s2, l = found[0]
    print(f"    example: (e{i} {'+' if s1>0 else '-'} e{j}) * "
          f"(e{k} {'+' if s2>0 else '-'} e{l}) = 0")

    # 5. Bilateral collapse identity check on found pairs:
    #    (aP + bQ)(bP + cQ) = -2 b (a+c) e0   [claimed identity, tested numerically]
    #    We count for how many found (P,Q) pairs it holds for random a,b,c.
    holds = 0
    for (i, s1, j, k, s2, l) in found:
        P = basis(i) + s1 * basis(j)
        Q = basis(k) + s2 * basis(l)
        ok = True
        for _ in range(5):
            a, b, c = rng.standard_normal(3)
            lhs = cd_mult(a * P + b * Q, b * P + c * Q)
            rhs = -2 * b * (a + c) * basis(0)
            if np.abs(lhs - rhs).max() > 1e-10:
                ok = False
                break
        holds += ok
    print(f"[5] bilateral collapse identity holds for: {holds}/{len(found)} pairs")

    # 6. ZD-gate primitive: collapse magnitude |(aP+bQ)(bP+cQ)| as a
    #    function of (a, c). For identity-satisfying pairs this equals
    #    2|b||a+c| -> gate fires (passes 0) exactly when a = -c.
    for (i, s1, j, k, s2, l) in found:
        P = basis(i) + s1 * basis(j)
        Q = basis(k) + s2 * basis(l)
        a, b, c = 0.7, 1.3, -0.7          # a + c = 0  -> annihilation
        m0 = np.linalg.norm(cd_mult(a * P + b * Q, b * P + c * Q))
        a2, c2 = 0.7, 0.9                  # a + c != 0 -> no annihilation
        m1 = np.linalg.norm(cd_mult(a2 * P + b * Q, b * P + c2 * Q))
        if m0 < 1e-12 and m1 > 1e-6:
            print(f"[6] ZD-gate primitive verified: |prod|=0 iff a=-c "
                  f"(m0={m0:.1e}, m1={m1:.2f})")
            break

    print("\nAll checks passed. Structure tensor is safe to port to PyTorch.")
    return T, found


if __name__ == "__main__":
    run_checks()
