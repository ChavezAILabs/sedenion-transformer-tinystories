"""verify_phase4_reggiani.py -- repo-side reproduction of the claims used
from Reggiani, "The geometry of sedenion zero divisors" (arXiv:2411.18881,
v5 2024-11-26), against this repo's sedenion_kernel.py.

Source of truth for the transcription: the paper's own LaTeX source
(zero-divisors-v5-2024-11-26.tex, downloaded from
https://arxiv.org/e-print/2411.18881), not a rendered PDF or an AI
summary of one -- Table 1 (the "84 standard zero divisors", Appendix
A.1, lines 504-533 of that file) is copied verbatim here, index by
index, sign by sign.

What this proves, each independently checkable:
  [1] All 84 published pairs are exact zero divisors under this repo's
      cd_mult (sanity: repo kernel convention matches the paper's).
  [2] The project's ZD_PAIR ((e3+e12, e5+e10), zda_layers.py/zda_grid.py)
      is literally row 43 of the paper's own table -- the project's
      "Pattern 2" choice is not just numerically verified in isolation,
      it is one of Reggiani's canonical 84.
  [3] The repo's own exhaustive two-blade search (sedenion_kernel.py
      check [4], and phase4_shuffledT_nulls.py's "84 null left-dirs")
      is independently reproduced from Reggiani's Prop 2.1 characterization
      (a, b imaginary octonions, equal nonzero norm, orthogonal) by direct
      combinatorial count, not just by exhaustive search -- explaining
      *why* the number is 84, not just that it is.
  [4] 336 = 84 x 4 is explained: dim(ann u) = 4 for every sedenion zero
      divisor u (Moreno's mod-4 result + Biss-Dugger-Isaksen's upper
      bound 2^n - 4n + 4 = 4 at n=4 pin it exactly, not just bound it),
      so each of the 84 canonical zero-divisor two-blades has exactly
      4 signed two-blade partners among the 210 candidates.

This is a repo-verification gate, run before Reggiani's results are used
in any pre-registration decision (ZDA_phase4_spec.md SS9 item 1, the
X-inequivalence certificate) -- see PHASE4_reggiani_reading.md.

Run:  zda-env\\Scripts\\python.exe verify_phase4_reggiani.py
"""
import numpy as np

from sedenion_kernel import cd_mult, structure_tensor

# ---------------------------------------------------------------
# [1]+[2] Table 1 transcription (84 pairs), verbatim from the LaTeX
# source, lines 510-530. Each tuple: (i, j, sign_j, k, l, sign_l) encodes
# the pair (e_i + sign_j*e_j,  e_k + sign_l*e_l).
# ---------------------------------------------------------------
TABLE = [
    (1,10,'+',4,15,'-'), (1,10,'+',5,14,'+'), (1,10,'+',6,13,'-'), (1,10,'+',7,12,'+'),
    (1,11,'+',4,14,'+'), (1,11,'+',5,15,'+'), (1,11,'+',6,12,'-'), (1,11,'+',7,13,'-'),
    (1,12,'+',2,15,'+'), (1,12,'+',3,14,'-'), (1,12,'+',6,11,'+'), (1,12,'+',7,10,'-'),
    (1,13,'+',2,14,'-'), (1,13,'+',3,15,'-'), (1,13,'+',6,10,'+'), (1,13,'+',7,11,'+'),
    (1,14,'+',2,13,'+'), (1,14,'+',3,12,'+'), (1,14,'+',4,11,'-'), (1,14,'+',5,10,'-'),
    (1,15,'+',2,12,'-'), (1,15,'+',3,13,'+'), (1,15,'+',4,10,'+'), (1,15,'+',5,11,'-'),
    (2,9,'+',4,15,'+'),  (2,9,'+',5,14,'-'),  (2,9,'+',6,13,'+'),  (2,9,'+',7,12,'-'),
    (2,11,'+',4,13,'-'), (2,11,'+',5,12,'+'), (2,11,'+',6,15,'+'), (2,11,'+',7,14,'-'),
    (2,12,'+',3,13,'+'), (2,12,'+',5,11,'-'), (2,12,'+',7,9,'+'),  (2,13,'+',3,12,'-'),
    (2,13,'+',4,11,'+'), (2,13,'+',6,9,'-'),  (2,14,'+',3,15,'-'), (2,14,'+',5,9,'+'),
    (2,14,'+',7,11,'+'), (2,15,'+',3,14,'+'), (2,15,'+',4,9,'-'),  (2,15,'+',6,11,'-'),
    (3,9,'+',4,14,'-'),  (3,9,'+',5,15,'-'),  (3,9,'+',6,12,'+'),  (3,9,'+',7,13,'+'),
    (3,10,'+',4,13,'+'), (3,10,'+',5,12,'-'), (3,10,'+',6,15,'-'), (3,10,'+',7,14,'+'),
    (3,12,'+',5,10,'+'), (3,12,'+',6,9,'-'),  (3,13,'+',4,10,'-'), (3,13,'+',7,9,'-'),
    (3,14,'+',4,9,'+'),  (3,14,'+',7,10,'-'), (3,15,'+',5,9,'+'),  (3,15,'+',6,10,'+'),
    (4,9,'+',6,11,'-'),  (4,9,'+',7,10,'+'),  (4,10,'+',5,11,'+'), (4,10,'+',7,9,'-'),
    (4,11,'+',5,10,'-'), (4,11,'+',6,9,'+'),  (4,13,'+',6,15,'+'), (4,13,'+',7,14,'-'),
    (4,14,'+',5,15,'-'), (4,14,'+',7,13,'+'), (4,15,'+',5,14,'+'), (4,15,'+',6,13,'-'),
    (5,9,'+',6,10,'-'),  (5,9,'+',7,11,'-'),  (5,10,'+',6,9,'+'),  (5,11,'+',7,9,'+'),
    (5,12,'+',6,15,'-'), (5,12,'+',7,14,'+'), (5,14,'+',7,12,'-'), (5,15,'+',6,12,'+'),
    (6,10,'+',7,11,'-'), (6,11,'+',7,10,'+'), (6,12,'+',7,13,'-'), (6,13,'+',7,12,'+'),
]
assert len(TABLE) == 84, f"transcription error: got {len(TABLE)} rows, expected 84"

def elt(i, si, j):
    v = np.zeros(16)
    v[i] = 1.0
    v[j] = 1.0 if si == '+' else -1.0
    return v

n_zero = 0
zd_pair_row = None
for idx, (i, j, si, k, l, sk) in enumerate(TABLE):
    prod = cd_mult(elt(i, si, j), elt(k, sk, l))
    ok = np.abs(prod).max() < 1e-12
    n_zero += ok
    if not ok:
        print(f"  MISMATCH row {idx}: (e{i}{si}e{j}, e{k}{sk}e{l}) -> |uv| = {np.abs(prod).max():.3e}")
    if (i, j, si) == (3, 12, '+') and (k, l, sk) == (5, 10, '+'):
        zd_pair_row = idx

print(f"[1] {n_zero}/84 published Table-1 pairs are exact zero divisors under repo cd_mult")
assert n_zero == 84

print(f"[2] project ZD_PAIR (e3+e12, e5+e10) == Table 1 row {zd_pair_row}: {zd_pair_row is not None}")
assert zd_pair_row is not None

# ---------------------------------------------------------------
# [3] Reconstruct the 84 from Prop 2.1 directly (not from the table):
# (a,b) is a zero divisor iff a,b are imaginary octonions, ||a||=||b||!=0,
# a perp b. For a signed two-blade e_i + s*e_j (i<j, i,j in 1..15) to be
# such a pair, i must sit in the first CD-half (1..7, i.e. "a") and j in
# the second CD-half excluding its real unit (9..15, i.e. "b"'s imaginary
# part) -- else one of a,b is zero and the norms can't match. Within
# that, a=e_i, b=e_{j-8} must be orthogonal as octonion imaginary units:
# i != j-8.
# ---------------------------------------------------------------
combos = [(i, j) for i in range(1, 8) for j in range(9, 16) if i != j - 8]
n_combinatorial = len(combos) * 2   # x2 for the sign on e_j
print(f"[3] Prop 2.1 combinatorial count: {len(combos)} (i,j) x 2 signs = {n_combinatorial}")
assert n_combinatorial == 84

# cross-check against the repo's own exhaustive search (sedenion_kernel / phase4_shuffledT_nulls)
T16 = structure_tensor(16)
verified_zd_blades = set()
for (i, j) in combos:
    for s in ('+', '-'):
        u = elt(i, s, j)
        A = np.einsum("mij,i->mj", T16, u)     # left-mult matrix for u
        smin = np.linalg.svd(A, compute_uv=False)[-1]
        if smin < 1e-10:
            verified_zd_blades.add((i, j, s))
print(f"    of these, {len(verified_zd_blades)}/84 have a singular left-mult matrix "
      f"(i.e. are genuine sedenion zero divisors, confirmed independently of Prop 2.1's "
      f"algebraic argument)")
assert len(verified_zd_blades) == 84

# ---------------------------------------------------------------
# [4] 336 = 84 x 4: each of the 84 canonical blades has exactly 4 signed
# two-blade partners (among all 210 candidates i<j in 1..15) that it
# annihilates -- consistent with dim(ann u) = 4 pinned exactly (not just
# bounded) by Moreno's mod-4 result + Biss-Dugger-Isaksen's n=4 bound.
# ---------------------------------------------------------------
def two_blades_full():
    V = []
    for i in range(1, 16):
        for j in range(i + 1, 16):
            for s in (1.0, -1.0):
                v = np.zeros(16); v[i] = 1.0; v[j] = s
                V.append(v)
    return np.array(V)

B = two_blades_full()
prods = np.einsum("mij,ai,bj->abm", T16, B, B)
null_mask = np.abs(prods).max(axis=2) < 1e-12
partner_counts = null_mask.sum(axis=1)   # per left-blade, how many right partners annihilate it
nonzero_partner_counts = partner_counts[partner_counts > 0]
print(f"[4] partner-count per canonical ZD blade: "
      f"min={nonzero_partner_counts.min()}, max={nonzero_partner_counts.max()} "
      f"(all should be 4); total pairs = {null_mask.sum()} (should be 336 = 84x4)")
assert (nonzero_partner_counts == 4).all()
assert null_mask.sum() == 336

print("\nAll Reggiani cross-checks passed. Table 1, Prop 2.1's combinatorics, and the "
      "84x4=336 structure are mutually consistent with this repo's kernel.")
