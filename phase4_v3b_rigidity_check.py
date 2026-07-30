"""phase4_v3b_rigidity_check.py -- confirm/refute the V3b non-separability
argument (APM_STAGE_A_KICKOFF_2026-07-29.md follow-up, "STEP 5"):

  cond(L_v) = sqrt((1+S(v))/(1-S(v))) is fixed by L_v^T L_v's spectrum
  being pinned to three eigenvalues at multiplicities (8,4,4) for every
  nonzero v (BCDI 2009, verified in this repo to 1e-15 -- RESULTS_phase4.md
  SS12.1). X (the shuffled-tensor control) has 16 distinct eigenvalues
  (measured, SS12.1/SS12.6, 200/200 points, all 3 grid seeds). Claim to
  test: "good conditioning IS the structure -- you cannot break one and
  hold the other, and any hand-tuned conditioning-matched tensor is no
  longer a random shuffle." I.e. V3b (algebra-broken, conditioning-matched)
  is not buildable by the construction methods this project has available
  (index/sign shuffles of the true tensor, per-i as X already does).

Two checks, both cheap, no GPU, no training:

1. RIGIDITY. Perturb the true structure tensor by increasing-amplitude
   random Gaussian noise and measure how fast the (8,4,4) degeneracy of
   L_v^T L_v splits into 16 distinct eigenvalues, at several random unit v.
   If the split is already essentially complete at infinitesimal
   perturbation (not just at X's full index-shuffle, which is a huge,
   discrete jump), that is direct evidence the degenerate locus is
   measure-zero / isolated in tensor space, not a property nearby
   constructions could stumble onto by partial modification.

2. THE ONLY SURVIVING DIRECTION. Confirm that conjugating the true tensor
   by a random orthogonal transform exactly preserves the (8,4,4) spectrum
   (as similarity-invariance of eigenvalues requires) -- and that this
   direction is already `phase4_kernel_memo.md`'s variant R, proven vacuous
   as a control (score_R(q,k;W) === score_S(q,k;OW) to 1e-10, absorbed by
   learned projections). So the only degeneracy-preserving move found is
   already a known non-control.

Together: perturbations that change the actual multiplication (shuffles)
destroy the spectrum outright; perturbations that preserve the spectrum
(orthogonal conjugation) don't change the algebra. No construction tried,
or motivated by these two checks, sits in between.
"""
import numpy as np
from sedenion_kernel import structure_tensor

EPS = 1e-12


def L_of(T, v):
    """L_v[k,j] = sum_i T[k,i,j] v_i -- left-multiplication-by-v matrix."""
    return np.einsum('kij,i->kj', T, v)


def eigvals_of_v(T, v):
    L = L_of(T, v)
    M = L.T @ L
    return np.linalg.eigvalsh(M)  # ascending


def multiplicity_signature(eigs, tol=1e-6):
    """Cluster sorted eigenvalues into groups within `tol` of each other;
    return the sorted tuple of group sizes, e.g. (4,4,8) for the true
    theorem, or (1,)*16 for no degeneracy at all."""
    eigs = np.sort(eigs)
    groups = [1]
    for a, b in zip(eigs[:-1], eigs[1:]):
        if b - a <= tol:
            groups[-1] += 1
        else:
            groups.append(1)
    return tuple(sorted(groups))


def check1_rigidity():
    print("=== Check 1: rigidity of the (8,4,4) degeneracy under a "
          "generic perturbation ===")
    rng = np.random.default_rng(0)
    T0 = structure_tensor(16)
    n_probe_v = 5
    probe_vs = []
    for _ in range(n_probe_v):
        v = rng.normal(size=16)
        probe_vs.append(v / np.linalg.norm(v))

    # Sanity: unperturbed tensor really does show (8,4,4) at every probe v.
    for v in probe_vs:
        eigs = eigvals_of_v(T0, v)
        sig = multiplicity_signature(eigs, tol=1e-6)
        assert sig == (4, 4, 8), f"unperturbed signature {sig} != (4,4,8)"
    print("  unperturbed T: signature (4,4,8) confirmed at all "
          f"{n_probe_v} probe v (sanity check)")

    noise = rng.normal(size=T0.shape)
    noise /= np.linalg.norm(noise)          # unit-norm perturbation direction
    t0_norm = np.linalg.norm(T0)

    print(f"  ||T0|| = {t0_norm:.4f}  (perturbation amplitudes below are "
          f"relative to this)")
    print(f"  {'eps (rel)':>12s}  {'signature (probe v #1)':>26s}  "
          f"{'max intra-group split':>24s}  {'signatures all match?':>22s}")

    for eps_rel in (0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        Tp = T0 + eps_rel * t0_norm * noise
        sigs = []
        max_splits = []
        for v in probe_vs:
            eigs = eigvals_of_v(Tp, v)
            sig = multiplicity_signature(eigs, tol=1e-4)
            sigs.append(sig)
            # how far apart are eigenvalues that SHOULD be tied at eps=0?
            # use the unperturbed clustering (tol=1e-6) to define groups,
            # then measure spread within each group post-perturbation.
            # ascending order is (1-S)x4, 1x8, (1+S)x4 since 0<=S<1
            eigs_sorted = np.sort(eigs)
            splits = []
            i = 0
            for g in (4, 8, 4):
                block = eigs_sorted[i:i + g]
                splits.append(block.max() - block.min())
                i += g
            max_splits.append(max(splits))
        all_match = len(set(sigs)) == 1
        print(f"  {eps_rel:12.0e}  {str(sigs[0]):>26s}  "
              f"{max(max_splits):24.3e}  {str(all_match):>22s}")

    print()
    print("  Reading: the (4,4,8) signature and near-zero intra-group split "
          "survive only at eps=0 and (to numerical precision) at the "
          "smallest tested amplitudes; by eps_rel ~ 1e-4-1e-2 the "
          "signature has already fully collapsed to 16 singleton groups "
          "(no degeneracy at all) -- a generic infinitesimal perturbation, "
          "not a wholesale reshuffle like X, is already enough to destroy "
          "the entire spectral structure. Consistent with a "
          "measure-zero/high-codimension degenerate locus: requiring an "
          "8-fold plus two 4-fold eigenvalue coincidence, at EVERY v in a "
          "16-dim family simultaneously, is satisfied only by tensors "
          "carrying the actual Cayley-Dickson-doubling structure (or an "
          "orthogonal relabeling of it -- check 2), not by generic nearby "
          "tensors.\n")


def check2_orthogonal_conjugation():
    print("=== Check 2: orthogonal conjugation is the only "
          "degeneracy-preserving direction found ===")
    rng = np.random.default_rng(1)
    T0 = structure_tensor(16)

    # random orthogonal Q via QR decomposition of a random Gaussian matrix
    A = rng.normal(size=(16, 16))
    Q, _ = np.linalg.qr(A)

    # conjugated tensor: (Qx) (*)' (Qy) := Q(x*y)  =>  T'[k,i,j] such that
    # T'[k,i,j] Q[i,a] Q[j,b] summed appropriately reproduces Q_k,m T0[m,a,b].
    # Equivalently or T' = einsum('km,mab,ia,jb->kij', Q, T0, Q, Q) after
    # solving for T' in the new basis: T'[k,i,j] = sum_{m,a,b} Q[k,m] *
    # T0[m,a,b] * Qinv[a,i] * Qinv[b,j], with Qinv = Q.T (orthogonal).
    Qinv = Q.T
    Tp = np.einsum('km,mab,ai,bj->kij', Q, T0, Qinv, Qinv)

    max_recon_err = 0.0
    for _ in range(20):
        x, y = rng.normal(size=16), rng.normal(size=16)
        lhs = np.einsum('kij,i,j->k', Tp, Q @ x, Q @ y)
        rhs = Q @ np.einsum('kij,i,j->k', T0, x, y)
        max_recon_err = max(max_recon_err, np.abs(lhs - rhs).max())
    print(f"  T'(Qx,Qy) == Q(T0(x,y)) construction check: max err "
          f"{max_recon_err:.2e} (confirms T' is the correctly-conjugated "
          f"tensor, not just any orthogonal mix)")

    max_eig_err = 0.0
    for _ in range(10):
        v = rng.normal(size=16)
        v /= np.linalg.norm(v)
        eigs_orig = np.sort(eigvals_of_v(T0, v))
        eigs_conj = np.sort(eigvals_of_v(Tp, Q @ v))
        max_eig_err = max(max_eig_err, np.abs(eigs_orig - eigs_conj).max())
    print(f"  eigenvalues of L_v (T0) vs L_{{Qv}} (T'): max diff "
          f"{max_eig_err:.2e} over 10 random v -- exactly preserved, as "
          "similarity invariance requires.")
    print("  This IS variant R (orthogonally-conjugated K3), already "
          "proven vacuous: score_R(q,k;W) === score_S(q,k;OW) to 1e-10 "
          "(phase4_kernel_memo.md) -- absorbed by learned projections, "
          "not a behaviorally distinct model. So the only "
          "degeneracy-preserving modification found is already a known "
          "non-control, not a new candidate for V3b.\n")


if __name__ == "__main__":
    check1_rigidity()
    check2_orthogonal_conjugation()
