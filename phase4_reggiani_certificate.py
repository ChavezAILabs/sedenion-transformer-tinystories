"""phase4_reggiani_certificate.py -- the X-inequivalence certificate that
PRIOR_ART_REVIEW_zda.md SS3.1(a) called for, built on Reggiani's theorem
(arXiv:2411.18881, repo-verified in verify_phase4_reggiani.py).

Reggiani proves Z(S) = {(u,v) in S x S : |u|=|v|=sqrt2, uv=0} is
*diffeomorphic to G2*, a smooth compact 14-dimensional manifold. A smooth
submanifold has full-rank normal Jacobian at *every* point, no exceptions.
That is a sharp, checkable prediction distinct from anything
phase4_shuffledT_nulls.py tested (which counts nulls and measures
reachability via random-direction sigma_min(A_x), but never asks whether
the null set itself is smooth):

  At any (u,v) with uv=0, let F(u,v) = uv (the bilinear product). Its
  differential restricted to the tangent space of the sphere product
  S^15 x S^15 at (u,v) -- i.e. to {du perp u} x {dv perp v}, 30 real dims
  -- is the 16x30 matrix J = [ R_v |_{u^perp}  ,  L_u |_{v^perp} ], where
  R_v(du) = du*v and L_u(dv) = u*dv (via the structure tensor). By the
  implicit function theorem, if J has full row rank 16, the null set is
  *locally a smooth 14-manifold* at (u,v); if rank(J) < 16, (u,v) is a
  singular point (locally higher-dimensional or non-manifold).

Prediction: for the TRUE T16 tensor, sigma_min(J) is bounded well away
from 0 at *every* one of the 336 known null pairs (Reggiani's theorem
forces this -- Z(S) has no singular points, being diffeomorphic to a Lie
group). For the shuffled tensor X, no such theorem holds; if any of its
accidental null pairs show near-zero sigma_min(J), those are genuine
singularities -- a geometric (not just statistical) sense in which X's
zero-divisor structure is not equivalent to the sedenions', independent
of and sharper than the raw null-count gap already on record.

This is additive to, not a replacement for, phase4_shuffledT_nulls.py
(already SS9 item 1 DONE) -- that script's count/reachability numbers
stand; this one adds the smoothness axis Reggiani's theorem makes
available. Proposal, not yet folded into the spec -- see
PHASE4_reggiani_reading.md SS4 for the disposition.

Run:  zda-env\\Scripts\\python.exe phase4_reggiani_certificate.py
"""
import numpy as np

from sedenion_kernel import structure_tensor

T16 = structure_tensor(16)


def shuffled_tensor(seed):
    """Verbatim copy of phase4_shuffledT_nulls.shuffled_tensor -- kept
    local so this script has no import-order dependency on that one."""
    g = np.random.default_rng(seed)
    L = np.zeros((16, 16, 16))
    L[0] = np.eye(16)
    for k in range(1, 16):
        perm = g.permutation(16)
        signs = g.integers(0, 2, 16) * 2 - 1
        L[k, np.arange(16), perm] = signs
    return np.transpose(L, (1, 0, 2))


def two_blades():
    V = []
    for i in range(1, 16):
        for j in range(i + 1, 16):
            for s in (1.0, -1.0):
                v = np.zeros(16); v[i] = 1.0; v[j] = s
                V.append(v)
    return np.array(V)


B = two_blades()   # (210, 16)


def find_null_pairs(T):
    """All (u,v) two-blade pairs with uv=0 under structure tensor T."""
    prods = np.einsum("mij,ai,bj->abm", T, B, B)
    mask = np.abs(prods).max(axis=2) < 1e-12
    idx = np.argwhere(mask)
    return [(B[a], B[b]) for a, b in idx]


def orthogonal_complement_projector(u):
    """Projector onto {x : x.u = 0} for unit vector u."""
    u = u / np.linalg.norm(u)
    return np.eye(len(u)) - np.outer(u, u)


def jacobian_sigma_min(T, u, v):
    """sigma_min of the 16x30 tangent-restricted Jacobian of F(u,v)=uv
    at a zero-divisor pair (u,v), under structure tensor T."""
    Rv = np.einsum("mij,j->mi", T, v)          # right-mult-by-v: du -> du*v  (16x16)
    Lu = np.einsum("mij,i->mj", T, u)          # left-mult-by-u:  dv -> u*dv  (16x16)
    Pu = orthogonal_complement_projector(u)     # restrict du to u_perp
    Pv = orthogonal_complement_projector(v)     # restrict dv to v_perp
    J = np.concatenate([Rv @ Pu, Lu @ Pv], axis=1)   # (16, 32) but rank <=30 after projection
    sv = np.linalg.svd(J, compute_uv=False)
    return sv[-1] if len(sv) >= 16 else 0.0   # 16th singular value = sigma_min for a 16-row matrix


def characterize_smoothness(T, label, max_pairs=None):
    pairs = find_null_pairs(T)
    if max_pairs is not None:
        rng = np.random.default_rng(2026)
        idx = rng.choice(len(pairs), size=min(max_pairs, len(pairs)), replace=False)
        pairs = [pairs[i] for i in idx]
    smins = np.array([jacobian_sigma_min(T, u, v) for (u, v) in pairs])
    n_singular = int((smins < 1e-6).sum())
    print(f"{label:<14} n_pairs={len(pairs):>4}  sigma_min(J): "
          f"min={smins.min():.4e}  median={np.median(smins):.4f}  max={smins.max():.4f}  "
          f"| singular points (sigma_min<1e-6): {n_singular}/{len(pairs)}")
    return smins, n_singular


if __name__ == "__main__":
    print("Reggiani smoothness certificate: sigma_min of the tangent-restricted\n"
          "Jacobian of F(u,v)=uv at every known null pair. Full rank (sigma_min\n"
          "bounded away from 0) == locally a smooth 14-manifold point (theorem-\n"
          "predicted for the true tensor at every point; not guaranteed for X).\n")

    smins_true, n_sing_true = characterize_smoothness(T16, "TRUE T16")
    assert n_sing_true == 0, (
        "Reggiani's theorem predicts NO singular points on Z(S) -- "
        "a violation here means either a repo kernel bug or a numerical-"
        "tolerance issue, not a real counterexample to a published theorem."
    )

    print()
    results = {}
    for seed in (1337, 1338, 1339):
        smins_x, n_sing_x = characterize_smoothness(shuffled_tensor(seed), f"X seed {seed}")
        results[seed] = (smins_x, n_sing_x)

    print("\nInterpretation:")
    print(f"  TRUE T16: {n_sing_true}/336 singular points (theorem-consistent: must be 0)")
    for seed, (smins_x, n_sing_x) in results.items():
        frac = n_sing_x / len(smins_x)
        verdict = ("has genuine singularities -- geometrically inequivalent to Z(S), "
                    "independent of the null-count gap" if n_sing_x > 0 else
                    "no singularities found at this sample -- smoothness alone does not "
                    "distinguish this draw from the true tensor; the count/reachability "
                    "gap (phase4_shuffledT_nulls.py) remains the operative distinction")
        print(f"  X seed {seed}: {n_sing_x}/{len(smins_x)} singular points ({frac:.1%}); {verdict}")
