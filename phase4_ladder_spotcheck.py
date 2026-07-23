"""phase4_ladder_spotcheck.py — spot-check (b) of the spec v1.0 residual
verifications (first run + PASS 2026-07-22, kept rerunnable per the
project rule that verifications get saved as scripts): the R_8
relative-position property at the SIX FROZEN ladder frequencies
(spec v1.0 SS2: omega_h = (2*pi/1024)**(h/(H-1)), H=6), with integer
positions spanning the 1024 length-gen context. Expected pass — the
property is frequency-agnostic per head (phase4_positional.py proves it
for R_8 at arbitrary phases); this pins it at the frozen values.

Checks per omega_h:
  B. |R_8(w*i)q (x) R_8(w*j)k| depends only on i-j  (rel-prop)
  C. shared phase preserves the Pattern-2 null at w*t
"""
import sys
import numpy as np

sys.path.insert(0, r"C:\dev\projects\apm-agi_tests")
from sedenion_kernel import basis, structure_tensor

T16 = structure_tensor(16)
L8 = T16[:, 8, :]
P = basis(3) + basis(12)
Q = basis(5) + basis(10)

def mult(x, y): return np.einsum("kij,i,j->k", T16, x, y)
def R8(th):     return np.cos(th) * np.eye(16) + np.sin(th) * L8

a0, b0 = 0.8, 0.5
x_null = a0 * P + b0 * Q
y_null = b0 * P - a0 * Q
assert np.linalg.norm(mult(x_null, y_null)) < 1e-12

H = 6
ladder = [(2 * np.pi / 1024) ** (h / (H - 1)) for h in range(H)]
assert abs(ladder[0] - 1.0) < 1e-15
assert abs(ladder[-1] - 2 * np.pi / 1024) < 1e-15
print("frozen ladder omega_h:", ", ".join(f"{w:.6f}" for w in ladder))

rng = np.random.default_rng(7)
qks = [(rng.standard_normal(16), rng.standard_normal(16)) for _ in range(5)]
pos_pairs = [(0, 1), (3, 7), (17, 100), (255, 256), (512, 1023)]
shifts = [1, 13, 511]

ok = True
for h, w in enumerate(ladder):
    worst_rel = 0.0
    for q, k in qks:
        for i, j in pos_pairs:
            base = np.linalg.norm(mult(R8(w * i) @ q, R8(w * j) @ k))
            for d in shifts:
                shifted = np.linalg.norm(
                    mult(R8(w * (i + d)) @ q, R8(w * (j + d)) @ k))
                worst_rel = max(worst_rel, abs(base - shifted))
    worst_null = max(
        np.linalg.norm(mult(R8(w * t) @ x_null, R8(w * t) @ y_null))
        for t in range(0, 1024, 37))
    rel_pass = worst_rel < 1e-9
    null_pass = worst_null < 1e-9
    ok &= rel_pass and null_pass
    print(f"h={h} omega={w:.6f}  rel-prop worst dev={worst_rel:.2e} "
          f"[{'PASS' if rel_pass else 'FAIL'}]  shared-phase null "
          f"worst={worst_null:.2e} [{'PASS' if null_pass else 'FAIL'}]")

print("\nSpot-check (b):", "PASS — rel-prop and null preservation hold at "
      "all six frozen frequencies" if ok else "FAIL")
sys.exit(0 if ok else 1)
