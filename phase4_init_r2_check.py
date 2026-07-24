"""phase4_init_r2_check.py -- N2: does r^2 at ACTUAL model initialization
(real K3Attention, real wq/wk projections, real frozen ladder rotation,
real grid dims) resemble free-sphere sampling of x,y, or is it confined
differently?

This is the question REPLY_to_ClaudeCode_2026-07-24.md / FINDINGS_...md
raised: their sigma_min accessibility numbers are for RAW unit vectors
x,y drawn uniformly from S^15 -- but the model never sees raw sphere
points, it sees q=W_q(embedding) then R_8-rotated. If that
parameterization doesn't sample freely from the sphere, free-sphere
stats don't directly explain what min(r^2) looks like in the real model
at step 0, and the "X already sits near its own variety" claim needs
this check before it can be applied to session-close S3(e)'s numbers.

Uses the REAL grid dims (phase4_grid.GRID_DIMS): d_model=384, n_heads=6,
ctx=256 -- not toy dims. Multiple independent random inits (simulating
different training-run seeds) per variant.
"""
import sys
sys.path.insert(0, r"C:\dev\projects\apm-agi_tests")
import numpy as np
import torch

from phase4_layers import K3Attention, shuffled_structure_tensor

D_MODEL, N_HEADS, CTX, BATCH = 384, 6, 256, 24

def r2_stats_at_init(tensor=None, seed_for_tensor=None, n_inits=5):
    """n_inits independent fresh K3Attention instantiations (simulating
    different training-run init seeds); for each, one batch of random
    input at ctx=256, report min/median/frac<thresholds of r2 over the
    whole causal-window batch, then aggregate across inits."""
    mins, medians, fracs_1e2, fracs_1e1 = [], [], [], []
    for init_seed in range(n_inits):
        torch.manual_seed(1000 + init_seed)
        m = K3Attention(D_MODEL, N_HEADS, causal=True,
                        tensor=tensor)
        x = torch.randn(BATCH, CTX, D_MODEL)
        with torch.no_grad():
            r2, _ = m.scores(x)   # (B,H,T,T)
        causal_ok = ~torch.triu(torch.ones(CTX, CTX, dtype=torch.bool), 1)
        vals = r2[:, :, causal_ok]   # (B,H,N_pairs)
        v = vals.flatten().numpy()
        mins.append(float(v.min()))
        medians.append(float(np.median(v)))
        fracs_1e2.append(float((v < 1e-2).mean()))
        fracs_1e1.append(float((v < 1e-1).mean()))
    return {
        "min_over_inits": min(mins),
        "min_per_init": [f"{x:.4f}" for x in mins],
        "median_of_medians": float(np.median(medians)),
        "frac<1e-2_mean": float(np.mean(fracs_1e2)),
        "frac<1e-1_mean": float(np.mean(fracs_1e1)),
    }

print(f"Grid dims: d_model={D_MODEL} n_heads={N_HEADS} ctx={CTX} batch={BATCH}")
print(f"(matches phase4_grid.GRID_DIMS -- the real anchor-grid config)\n")

print("=== S (true tensor), r^2 at fresh init, 5 independent inits ===")
s_stats = r2_stats_at_init(tensor=None, n_inits=5)
for k, v in s_stats.items():
    print(f"  {k}: {v}")

for seed in (1337, 1338, 1339):
    print(f"\n=== X seed={seed}, r^2 at fresh init, 5 independent inits ===")
    x_stats = r2_stats_at_init(tensor=shuffled_structure_tensor(seed), n_inits=5)
    for k, v in x_stats.items():
        print(f"  {k}: {v}")

print("\nCompare to session-close S3(e)'s ACTUAL step-0 min(r^2) at L0:")
print("  S seed1337=0.1529 S seed1338=0.1382  X seed1337=0.0522 X seed1338=0.0820")
print("\nReading: if this script's min_over_inits for S/X (fresh random init,")
print("no training) lands in the same ballpark as session-close's real step-0")
print("numbers, the model's actual init DOES resemble what a random-projection")
print("init produces, and free-sphere framing is a reasonable proxy for the")
print("baseline-asymmetry argument. If this script's numbers are wildly")
print("different (e.g. all near 0, or all near 1, regardless of variant),")
print("the model's parameterization confines the accessible region")
print("differently than free sphere-of-x,y sampling, and REPLY's baseline-")
print("asymmetry argument needs a different baseline computation, not this")
print("proxy.")
