"""phase4_matched_N_null.py -- Stage 0 "can't vs doesn't" matched-N
random-key null control (chat-side spec, 2026-07-27 handoff).

RESULTS_phase4.md SS12.2 found: at real init, S's real keys land near
their theoretical floor 40% of the time (median ratio 2.3x); X's floor is
~100-150x lower than S's but real keys never come close (0/200 within 2x).
Read naively that looks like "S steers toward its floor, X doesn't" -- but
S's floor sits on a 4-dim eigenspace (BCDI multiplicity-4 annihilator) and
X's on a 1-dim needle (measured annihilator dimension 1). A 1-dim target
is unreachable by N random draws almost regardless of any steering effect,
so the raw floor-ratio gap conflates "doesn't steer" with "target is too
thin for anyone to hit by chance." This script isolates the two.

Method: at each real-init sampled causal query point P (SAME methodology
as SS12.2 -- real K3Attention, real wq/wk, real R_8 rotation, real grid
dims d_model=384/n_heads=6/ctx=256, fresh untrained weights, 5 inits x 40
points per variant = 200 points), take the real keys actually present at
that causal position (N = position+1 of them) and also draw N freshly-
sampled random unit vectors (repeated over many trials to smooth single-
draw noise). Compare:

  ratio_null = (actual min r^2 among the N real keys)
             / (min r^2 among N random unit draws, matched N)

  ratio_null ~= 1 for a variant  -> its real keys are no better than N
                                     random guesses at finding a low-r^2
                                     partner -- "doesn't steer"
  ratio_null <  1                -> real keys beat chance -- "steers"

Decision rule (as specified):
  both ~= 1   -> neither steers; the SS12.2 gap is pure target-size
                 geometry, "doesn't" is NOT supported
  S<1, X~=1   -> S steers, X does not; "doesn't" survives
  both < 1    -> both steer; compare magnitudes

Reports BOTH median-of-ratios and ratio-of-medians explicitly (resolves
the ambiguity flagged against SS12.2's single "median ratio" column).
"""
import sys
sys.path.insert(0, r"C:\dev\projects\apm-agi_tests")
import numpy as np
import torch

from phase4_layers import K3Attention, shuffled_structure_tensor
from sedenion_kernel import structure_tensor

D_MODEL, N_HEADS, CTX, BATCH = 384, 6, 256, 24
N_ALG = 16
N_INITS = 5
POINTS_PER_INIT = 40
N_RANDOM_TRIALS = 100
EPS = 1e-12


def r2_of(T, P, Qs):
    """r2 = ||P (x) Q||^2 / (|P|^2 |Q|^2) for a batch of Q's.
    T: (16,16,16) numpy structure tensor (m,i,j).
    P: (16,) numpy. Qs: (N,16) numpy."""
    A = np.einsum('mij,i->mj', T, P)          # (16,16): A[m,j]
    prod = np.einsum('mj,nj->nm', A, Qs)      # (N,16)
    num = (prod ** 2).sum(-1)                 # (N,)
    den = (P ** 2).sum() * (Qs ** 2).sum(-1) + EPS
    return num / den


def achievable_floor(T, P):
    """min over unit Q of r2(P,Q) = min eigenvalue of A^T A / |P|^2."""
    A = np.einsum('mij,i->mj', T, P)
    M = (A.T @ A) / ((P ** 2).sum() + EPS)
    ev = np.linalg.eigvalsh(M)
    return float(ev.min())


def sample_points(rng, n):
    """n causal (batch, head, t) index tuples, t in [0, CTX-1] so that
    N = t+1 real keys exist (t=0 -> N=1)."""
    b = rng.integers(0, BATCH, n)
    h = rng.integers(0, N_HEADS, n)
    t = rng.integers(0, CTX, n)
    return list(zip(b.tolist(), h.tolist(), t.tolist()))


def run_variant(T_numpy, label, seed_for_rng):
    """5 fresh inits x 40 points = 200 points; for each, actual min r2
    among real keys, achievable floor, and matched-N random null."""
    rng = np.random.default_rng(seed_for_rng)
    actual_mins, floors, null_mins, ratios, Ns = [], [], [], [], []

    for init_seed in range(N_INITS):
        torch.manual_seed(1000 + init_seed)
        m = K3Attention(D_MODEL, N_HEADS, causal=True, tensor=T_numpy)
        x = torch.randn(BATCH, CTX, D_MODEL)
        xf = x.float()
        with torch.no_grad():
            q = m.wq(xf).view(BATCH, CTX, N_HEADS, N_ALG).transpose(1, 2)
            k = m.wk(xf).view(BATCH, CTX, N_HEADS, N_ALG).transpose(1, 2)
            pos = torch.arange(CTX, dtype=torch.float32)
            q_rot = m._rotate(q, pos).numpy()   # (B,H,T,A)
            k_rot = m._rotate(k, pos).numpy()

        T = m.T.numpy()   # the actual tensor the module holds (S or X)
        points = sample_points(rng, POINTS_PER_INIT)
        for (b, h, t) in points:
            P = q_rot[b, h, t, :]
            N = t + 1
            real_keys = k_rot[b, h, :N, :]        # (N,16)
            actual_r2 = r2_of(T, P, real_keys)
            actual_min = float(actual_r2.min())

            floor = achievable_floor(T, P)

            trial_mins = np.empty(N_RANDOM_TRIALS)
            for trial in range(N_RANDOM_TRIALS):
                rand_Q = rng.normal(size=(N, N_ALG))
                rand_Q /= np.linalg.norm(rand_Q, axis=-1, keepdims=True) + EPS
                trial_mins[trial] = r2_of(T, P, rand_Q).min()
            random_N_min = float(np.median(trial_mins))

            actual_mins.append(actual_min)
            floors.append(floor)
            null_mins.append(random_N_min)
            ratios.append(actual_min / (random_N_min + EPS))
            Ns.append(N)

    actual_mins = np.array(actual_mins)
    floors = np.array(floors)
    null_mins = np.array(null_mins)
    ratios = np.array(ratios)
    Ns = np.array(Ns)

    print(f"=== {label} ===")
    print(f"  n points: {len(actual_mins)}  (N range {Ns.min()}-{Ns.max()}, "
          f"median N={int(np.median(Ns))})")
    print(f"  median actual min r2:      {np.median(actual_mins):.4f}")
    print(f"  median achievable floor:   {np.median(floors):.4f}")
    print(f"  median random-N-min r2:    {np.median(null_mins):.4f}")
    print(f"  ratio_of_medians (actual/random-N):  "
          f"{np.median(actual_mins) / np.median(null_mins):.4f}")
    print(f"  median_of_ratios (actual/random-N):   "
          f"{np.median(ratios):.4f}")
    print(f"  mean_of_ratios   (actual/random-N):   {np.mean(ratios):.4f}")
    print(f"  frac points with ratio_null < 0.5:    "
          f"{(ratios < 0.5).mean():.3f}")
    print(f"  frac points with ratio_null in [0.5,2]:"
          f"{((ratios >= 0.5) & (ratios <= 2.0)).mean():.3f}")
    print()
    return {
        "label": label,
        "median_actual": float(np.median(actual_mins)),
        "median_floor": float(np.median(floors)),
        "median_null": float(np.median(null_mins)),
        "ratio_of_medians": float(np.median(actual_mins) / np.median(null_mins)),
        "median_of_ratios": float(np.median(ratios)),
        "mean_of_ratios": float(np.mean(ratios)),
    }


if __name__ == "__main__":
    print(f"Grid dims: d_model={D_MODEL} n_heads={N_HEADS} ctx={CTX} "
          f"batch={BATCH}, {N_INITS} inits x {POINTS_PER_INIT} points, "
          f"{N_RANDOM_TRIALS} random trials per point\n")

    results = []
    results.append(run_variant(None, "S (true tensor)", seed_for_rng=42))
    for seed in (1337, 1338, 1339):
        results.append(run_variant(shuffled_structure_tensor(seed),
                                    f"X seed={seed}", seed_for_rng=seed))

    print("=== Summary table ===")
    print(f"{'variant':<14}{'actual':>10}{'floor':>10}{'randN':>10}"
          f"{'ratio_med':>12}{'med_ratio':>12}")
    for r in results:
        print(f"{r['label']:<14}{r['median_actual']:>10.4f}"
              f"{r['median_floor']:>10.4f}{r['median_null']:>10.4f}"
              f"{r['ratio_of_medians']:>12.4f}{r['median_of_ratios']:>12.4f}")
