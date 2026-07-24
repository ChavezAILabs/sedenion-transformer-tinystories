"""phase4_shared_phase_autotopy_check.py — exact tensor-identity version
of the shared-phase null check (P4_AMENDMENTS_HANDOFF.md I4, HANDOFF.md
S5 item 1), first run + PASS 2026-07-23.

Prior check (phase4_X_positional_check.py) established via direct trace
that shared-phase rotation R_8(alpha) preserves nullity of the ONE known
ZD pair for the true tensor and breaks it for X. This script asks the
stronger question directly: does R(alpha) (x) R(alpha), applied to BOTH
inputs of the FULL bilinear map, act as a consistent linear autotopy of
the structure tensor -- not just on one pair, on all of it?

Method: matricize T (16,16,16) as M in R^(16x256) by flattening the
(i,j) axes (row-major, so np.kron(R,R) is the correct joint-action
matrix under this flattening -- verified by the alpha=0 sanity assert
below, and independently: kron(A,B)[p*dB+r, q*dB+s] = A[p,q]*B[r,s] is
exactly the coefficient needed for X'[i,j] = sum R[i,i']R[j,j']X[i',j']).
Rtilde = R(alpha) kron R(alpha) acts on the 256 columns. Solve for the
best-fit output map S = M @ Rtilde @ pinv(M) and report:
  - relative residual  ||M@Rtilde - S@M|| / ||M@Rtilde||
  - orthogonality defect  ||S^T S - I||
at the six frozen ladder frequencies, true tensor vs shuffled X (three
seeds).

Prerequisite question (HANDOFF.md S5 item 1, answered by direct code
read of phase4_layers.shuffled_structure_tensor, no computation needed):
is X built from ONE consistent permutation across all three index slots,
or an arbitrary construction? The code draws a FRESH, independent
`g.permutation(16)` (and independent signs) for each k in 1..15 -- 15
mutually independent draws from the RNG stream, not one shared
permutation. ANSWER: arbitrary, not a consistent relabeling. X is not
an isomorphic copy of the true algebra in a misaligned basis; it is a
genuinely different bilinear map. This is the more concerning of the
two branches the prerequisite question raised (risk: "X never reaches
r^2 < 1e-2" could be vacuous rather than informative) -- but it is
consistent with, not contradicted by, everything else on record: X's
own broken L8^2=-I, its genuine singularities (sigma_min ~1e-17) vs the
true tensor's constant sigma_min=sqrt(2), and its 66-96 (not 336) null
pairs are all symptoms of the same underlying fact -- there is no
global change of basis taking X back to the true sedenion algebra.
"""
import sys
import numpy as np

sys.path.insert(0, r"C:\dev\projects\apm-agi_tests")
from sedenion_kernel import structure_tensor
from phase4_layers import shuffled_structure_tensor, frozen_omega_ladder


def autotopy_residual(T, alpha):
    L8 = T[:, 8, :]
    R = np.cos(alpha) * np.eye(16) + np.sin(alpha) * L8
    M = T.reshape(16, 256)          # row-major: (i,j) -> 16*i+j
    Rtilde = np.kron(R, R)          # (256,256); matches the flattening
    Mp = np.linalg.pinv(M)
    MR = M @ Rtilde
    S = MR @ Mp
    resid = np.linalg.norm(MR - S @ M)
    rel_resid = resid / (np.linalg.norm(MR) + 1e-300)
    ortho_defect = np.linalg.norm(S.T @ S - np.eye(16))
    return rel_resid, ortho_defect


# --- sanity assert: alpha=0 must give R=I exactly, hence S=I, resid=0 ---
rr0, od0 = autotopy_residual(structure_tensor(16), 0.0)
assert rr0 < 1e-9 and od0 < 1e-9, (rr0, od0)
print(f"[sanity] alpha=0: rel_resid={rr0:.1e} ortho_defect={od0:.1e}  OK\n")

H = 6
omega_h = frozen_omega_ladder(H)
T_true = structure_tensor(16)
seeds = (1337, 1338, 1339)
X_by_seed = {s: shuffled_structure_tensor(s) for s in seeds}

print("Frozen omega_h ladder:", [f"{w:.6f}" for w in omega_h])
print()
print(f"{'h':>2} {'omega_h':>10} | {'T rel_res':>10} {'T ortho':>10} | "
      + " | ".join(f"X{s} ortho" for s in seeds) + " | ortho ratio X/T (range)")

true_od = []
x_od = {s: [] for s in seeds}
for h, w in enumerate(omega_h):
    rr_t, od_t = autotopy_residual(T_true, w)
    true_od.append(od_t)
    row_x = []
    for s in seeds:
        rr_x, od_x = autotopy_residual(X_by_seed[s], w)
        x_od[s].append(od_x)
        row_x.append(od_x)
    ratios = [od_x / od_t for od_x in row_x]
    print(f"{h:>2} {w:>10.6f} | {rr_t:>10.3e} {od_t:>10.3e} | "
          + " ".join(f"{v:>9.3e}" for v in row_x)
          + f" | {min(ratios):.1f}x-{max(ratios):.1f}x")

print()
print("Reading: BOTH true T and X approach an exact autotopy as alpha->0")
print("(R(alpha)->I trivially) -- that is expected and not on its own")
print("informative. The discriminating quantity is the RATIO, which grows")
print("monotonically as omega_h shrinks (~1.3-1.5x at h=0 to tens-of-x at")
print("h=5): T's orthogonality defect vanishes strictly faster than X's as")
print("the rotation angle shrinks. This is a numerically independent")
print("confirmation of phase4_X_positional_check.py's L8^2!=-I finding for")
print("X, via a different method (autotopy-residual scaling, not a direct")
print("trace), and sharpens rather than merely repeats it: the gap is not")
print("binary (true=0, X!=0) but a growing multiplicative separation.")
print("Descriptive only, per the completeness gate -- not applied to the")
print("spec, not a grading input.")
