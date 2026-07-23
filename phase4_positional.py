"""phase4_positional.py — settles PHASE4_kernel_memo.md SS4 (the open
positional derivation) numerically against the repo tensor.

Candidate positional action: R_i(theta) = cos(theta) I + sin(theta) L_i,
with L_i = left-multiplication by basis e_i (orthogonal; memo check 11).
R_i is a one-parameter rotation group iff L_i^2 = -I.

Per generator i = 1..15:
  A. L_i^2 = -I (valid rotation group)?
  B. Relative-position property for the K3 kernel: does
     |R_i(th)q (x) R_i(ph)k| depend only on ph - th, for RANDOM q,k?
     (|R q| = |q| holds for free, so this makes the whole ratio r
     relative. Dot-product scores get this from orthogonality alone;
     the sedenion product need not cooperate.)
  C. Shared phase preserves nulls: |R_i(t)x (x) R_i(t)y| = 0 for an
     annihilating Pattern-2 pair, all t?
  D. Relative phase vs the manifold: |R_i(t)x (x) y| over t — identically
     0 (position never traverses the manifold) or oscillating (position
     moves pairs on/off the manifold: the memo's feature-or-confound)?
  E. Is the Pattern-2 plane span{P,Q} itself preserved?

Result (first run 2026-07-19): A holds for all 15. B holds ONLY for i=8
(the Cayley-Dickson doubling generator). For i=8, C and D are both
identically zero: e8-position is fully compatible with K3 AND never
traverses the annihilation manifold — position and ZD structure decouple
exactly. Every other generator fails B, and all but 6, 8, 14 break
shared-phase nulls; all except 8 oscillate under relative phase.
The Phase 4 positional action is therefore R_8, uniquely.
"""
import sys
import numpy as np

from sedenion_kernel import basis, structure_tensor

rng = np.random.default_rng(7)
T16 = structure_tensor(16)
L = [T16[:, i, :] for i in range(16)]          # (e_i * x)_m = L[i][m,j] x_j
P = basis(3) + basis(12)
Q = basis(5) + basis(10)

def mult(x, y): return np.einsum("kij,i,j->k", T16, x, y)
def R(i, th):   return np.cos(th) * np.eye(16) + np.sin(th) * L[i]

# annihilating Pattern-2 configuration (on-manifold): a + c = 0
a0, b0 = 0.8, 0.5
x_null = a0 * P + b0 * Q
y_null = b0 * P - a0 * Q
assert np.linalg.norm(mult(x_null, y_null)) < 1e-12

print(f"{'i':>3} {'L_i^2=-I':>9} {'rel-prop':>9} {'null@shared':>12} "
      f"{'null@rel(min/max)':>19} {'planePQ':>8}")
qks = [(rng.standard_normal(16), rng.standard_normal(16)) for _ in range(3)]
ths = np.linspace(0, 2 * np.pi, 25)

rows = []
for i in range(1, 16):
    sq = np.abs(L[i] @ L[i] + np.eye(16)).max() < 1e-12          # A
    worst = 0.0                                                   # B
    for q, k in qks:
        for th, ph in ((0.3, 1.1), (0.9, 2.0)):
            for d in (0.4, 1.3):
                n1 = np.linalg.norm(mult(R(i, th) @ q, R(i, ph) @ k))
                n2 = np.linalg.norm(mult(R(i, th + d) @ q, R(i, ph + d) @ k))
                worst = max(worst, abs(n1 - n2))
    rel = worst < 1e-9
    shared = max(np.linalg.norm(mult(R(i, t) @ x_null, R(i, t) @ y_null))
                 for t in ths)                                    # C
    relnorms = [np.linalg.norm(mult(R(i, t) @ x_null, y_null)) for t in ths]
    B_pq = np.stack([P / np.linalg.norm(P), Q / np.linalg.norm(Q)], axis=1)
    img = np.stack([L[i] @ P, L[i] @ Q], axis=1)                  # E
    plane = np.abs(img - B_pq @ (B_pq.T @ img)).max() < 1e-12
    rows.append((i, sq, rel, shared, min(relnorms), max(relnorms), plane))
    print(f"{i:>3} {str(sq):>9} {str(rel):>9} {shared:>12.2e} "
          f"{min(relnorms):>8.3f}/{max(relnorms):<8.3f} {str(plane):>8}")

nA = sum(r[1] for r in rows); nB = sum(r[2] for r in rows)
nC = sum(r[3] < 1e-9 for r in rows); nE = sum(r[6] for r in rows)
print(f"\nSummary: L_i^2=-I for {nA}/15; relative property holds for {nB}/15; "
      f"shared-phase null preserved for {nC}/15; plane preserved for {nE}/15.")
good = [r[0] for r in rows if r[2] and r[3] < 1e-9]
print("Generators with BOTH rel-prop and shared-phase null:", good or "none")
print("Generators traversing the manifold under relative phase "
      "(min~0 but max>0):",
      [r[0] for r in rows if r[4] < 1e-6 and r[5] > 1e-3])

assert good == [8], "expected e8 to be the unique compatible generator"
print("\nPhase 4 positional action: R_8 (unique). PASS")
