"""phase4_X_positional_check.py — does R_8's positional decoupling
(phase4_positional.py, proven for the TRUE structure tensor) carry over
to the shuffled tensor (variant X)? Run 2026-07-22 at the request of
PHASE4_session_close_2026-07-22.md SS5 item 1 ("the single highest-
value item on the list": test shared-phase null preservation for X).

phase4_positional.py never had to ask this: the true tensor's L_8 comes
from Cayley-Dickson structure and is proven (checks A/B/C/D there) to
give R_8(theta) a genuine rotation that decouples position from the
zero-divisor manifold exactly. X's L_8 is whatever
phase4_layers.shuffled_structure_tensor(seed) happens to produce at
index 8 — nothing guarantees it inherits any of that.

Result (all three grid seeds, reproduced identically in kind):
  [A] L_8^2 = -I FAILS for X (deviation exactly 1 or 2 in max-norm,
      seed-dependent) -- X's R_8(theta) is NOT an orthogonal rotation.
  Norm preservation is measurably violated (|R(theta)v| drifts from
      |v| by up to ~0.44 across the frozen H=6 ladder x pos 0..1023),
      not floating-point noise.
  Shared-phase null preservation is BROKEN for every X null pair, every
      frozen-ladder head, both seeds -- confirmed exactly null at t=0,
      then order-1 violations (~1.5-2.9, comparable to the null
      vectors' own norm sqrt(2)) within a few positions.

Interpretation (descriptive; NOT applied to the spec, NOT a grading
input — same status as the Reggiani smoothness certificate before it
was folded into SS9.1): sharpens, does not reverse, CLAUDE.md's existing
X caveat ("X is not null-matched... different, denser ZD-like
structure"). For X, position and (shuffled) ZD structure do NOT decouple
the way phase4_positional.py proves they do for the true tensor -- the
positional rotation itself is non-unitary for X, and it actively moves
X's own null pairs on and off its own manifold as an artifact of
absolute position, not just structure. Candidate mechanistic link to
PHASE4_session_close_2026-07-22.md SS3(e) (X's L0 never exhibits
manifold-seeking behavior, min r2 barely moves, frac r2<1e-2 stays
exactly 0 in both seeds): a manifold that isn't phase-stable under the
model's own positional code is a much harder "attend-here" target to
learn than one that provably is. Flagged as a candidate, not asserted
as the mechanism -- held for the same completeness-gate discipline as
everything else in Phase 4 until 18/18.
"""
import numpy as np

from phase4_layers import shuffled_structure_tensor, frozen_omega_ladder

N_ALG = 16


def mult(T, x, y):
    return np.einsum("kij,i,j->k", T, x, y)


def two_blade_null_pairs(T, tol=1e-12):
    V = []
    for i in range(1, 16):
        for j in range(i + 1, 16):
            for s in (+1.0, -1.0):
                v = np.zeros(16); v[i] = 1.0; v[j] = s
                V.append(v)
    V = np.array(V)
    prods = np.einsum("kij,ai,bj->abk", T, V, V)
    mask = np.abs(prods).max(axis=2) < tol
    return [(V[a], V[b]) for a, b in zip(*np.nonzero(mask))]


if __name__ == "__main__":
    ladder = frozen_omega_ladder(6)      # actual grid H=6 frequencies
    positions = np.arange(0, 1024, 7)    # dense coverage over L_MAX
    print(f"frozen ladder (H=6): {['%.6f' % w for w in ladder]}\n")

    for seed in (1337, 1338, 1339):
        T = shuffled_structure_tensor(seed)
        L8 = T[:, 8, :]

        A_dev = float(np.abs(L8 @ L8 + np.eye(16)).max())
        A_holds = A_dev < 1e-9
        print(f"=== X seed {seed} ===")
        print(f"  [A] L_8^2 = -I ? max|L_8^2 + I| = {A_dev:.3e}  "
              f"[{'HOLDS' if A_holds else 'FAILS'}]")

        def R(th, L8=L8):
            return np.cos(th) * np.eye(16) + np.sin(th) * L8

        rng = np.random.default_rng(seed + 1)
        v = rng.standard_normal(16)
        norm_dev = max(abs(np.linalg.norm(R(w * p) @ v) - np.linalg.norm(v))
                       for w in ladder for p in (0, 100, 1023))
        print(f"  norm preservation |R(th)v| vs |v|, worst dev: "
              f"{norm_dev:.3e}")

        pairs = two_blade_null_pairs(T)
        print(f"  {len(pairs)} exact two-blade null pairs (X's own)")
        if not pairs:
            print("  no pairs to test\n")
            continue

        worst_by_head = []
        for w in ladder:
            worst = 0.0
            for x_null, y_null in pairs:
                for t in positions:
                    worst = max(worst, float(np.linalg.norm(
                        mult(T, R(w * t) @ x_null, R(w * t) @ y_null))))
            worst_by_head.append(worst)
        print("  shared-phase null preservation, worst |mult(Rx,Ry)| "
              "per head:")
        for h, (w, worst) in enumerate(zip(ladder, worst_by_head)):
            status = "PRESERVED" if worst < 1e-9 else "BROKEN"
            print(f"    h={h} omega={w:.6f}  worst={worst:.3e}  "
                  f"[{status}]")
        overall = ("PRESERVED (all heads)" if max(worst_by_head) < 1e-9
                   else "BROKEN (at least one head)")
        print(f"  => {overall}\n")

    print("Result: all three seeds show [A] FAILS, norm preservation "
          "genuinely violated, shared-phase null preservation BROKEN "
          "at every head. See module docstring for the interpretation "
          "status (descriptive, not applied to the spec).")
