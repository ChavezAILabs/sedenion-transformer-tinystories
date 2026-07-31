"""phase4_priorityA_review_verify.py -- verifies the four confirmed points
and refutes the one dead point from PHASE5_priorityA_review_2026-07-31.md
(chat-side review of PHASE5_priorityA_crossing_2026-07-31.md), against
real source data rather than the review's own pooled-mean arithmetic.
Results are folded into the crossing document's 2026-07-31 revision; this
script is the rerunnable record of how they were checked.

No GPU required. ppl figures below are transcribed from
p4_priorityA_crossing_results.json (the actual Colab run output);
val_loss is read live from p4_artifacts/*/summary.json.
"""

import json
import math
import statistics as st

PPL = {
 "S":  {1337: {256: 6.061, 512: 12.137, 1024: 39.806, 2048: 86.379, 4096: 134.657},
        1338: {256: 6.123, 512: 11.216, 1024: 34.376, 2048: 73.848, 4096: 117.753},
        1339: {256: 6.125, 512: 10.75,  1024: 30.52,  2048: 64.522, 4096: 99.05}},
 "X":  {1337: {256: 5.895, 512: 18.665, 1024: 52.219, 2048: 93.173, 4096: 125.29},
        1338: {256: 5.908, 512: 18.045, 1024: 51.605, 2048: 91.058, 4096: 124.531},
        1339: {256: 5.937, 512: 17.959, 1024: 51.096, 2048: 93.28,  4096: 126.93}},
 "D1": {1337: {256: 5.198, 512: 8.673,  1024: 24.628, 2048: 53.595, 4096: 78.987,  8192: 98.594},
        1338: {256: 5.247, 512: 14.458, 1024: 35.061, 2048: 65.79,  4096: 89.804,  8192: 106.908},
        1339: {256: 5.238, 512: 16.904, 1024: 42.442, 2048: 75.928, 4096: 98.569,  8192: 112.902}},
 "D0": {1337: {256: 5.225, 512: 7.375,  1024: 18.684, 2048: 37.6,   4096: 61.695,  8192: 85.328},
        1338: {256: 5.246, 512: 7.817,  1024: 19.462, 2048: 40.638, 4096: 69.454,  8192: 97.645},
        1339: {256: 5.228, 512: 8.466,  1024: 23.923, 2048: 49.625, 4096: 81.426,  8192: 114.575}},
 "D0p": {1337: {256: 5.33,  512: 12.553, 1024: 35.055, 2048: 68.097, 4096: 94.89,   8192: 114.555},
         1338: {256: 5.357, 512: 6.89,   1024: 24.406, 2048: 54.927, 4096: 82.396,  8192: 102.659},
         1339: {256: 5.334, 512: 10.068, 1024: 27.239, 2048: 58.62,  4096: 85.593,  8192: 108.111}},
 "Q0": {1337: {256: 6.228, 512: 21.08,  1024: 45.496, 2048: 68.059, 4096: 83.493,  8192: 93.191},
        1338: {256: 6.287, 512: 20.962, 1024: 47.18,  2048: 71.99,  4096: 87.508,  8192: 98.035},
        1339: {256: 6.238, 512: 19.573, 1024: 44.965, 2048: 70.514, 4096: 86.323,  8192: 97.812}},
}
SEEDS = (1337, 1338, 1339)
CTXS = (256, 512, 1024, 2048, 4096)


def load_val_loss():
    out = {}
    for v in PPL:
        out[v] = {}
        for s in SEEDS:
            with open(f"p4_artifacts/{v}_seed{s}/summary.json") as f:
                out[v][s] = json.load(f)["final_val_loss"]
    return out


def check_1_normalized_crossing():
    """Review point 1: does the normalized metric (ppl / e^own-val-loss)
    also cross S vs D1? Rigorous version: normalize per-seed using that
    seed's OWN val_loss, then pool -- not pooled-ppl / pooled-val-loss."""
    vl = load_val_loss()
    print("=== Point 1: normalized metric, S vs D1 (per-seed, then pooled) ===")
    for c in CTXS:
        s_norm = [PPL["S"][s][c] / math.exp(vl["S"][s]) for s in SEEDS]
        d_norm = [PPL["D1"][s][c] / math.exp(vl["D1"][s]) for s in SEEDS]
        sm, dm = st.mean(s_norm), st.mean(d_norm)
        winner = "S" if sm < dm else "D1"
        print(f"  ctx{c}: S={sm:.3f} D1={dm:.3f}  winner={winner} "
              f"by {abs(sm - dm) / max(sm, dm) * 100:.1f}%")


def check_2_q0_rank():
    """Review point 2: does Q0's rank climb before ctx=8192?"""
    print("\n=== Point 2: Q0 rank by context (raw ppl, pooled mean) ===")
    for c in (256, 512, 1024, 2048, 4096, 8192):
        row = []
        for v in PPL:
            vals = [PPL[v][s].get(c) for s in SEEDS]
            if all(x is not None for x in vals):
                row.append((st.mean(vals), v))
        row.sort()
        rank = next(i + 1 for i, (_, v) in enumerate(row) if v == "Q0")
        print(f"  ctx{c}: {' < '.join(v for _, v in row)}   Q0 rank = {rank}/{len(row)}")


def check_1b_normalized_per_seed():
    """Follow-up to point 1: pooled-mean crossing hides that every rung from
    512-4096 is a 2-of-3 (or 1-of-3) seed split, never unanimous. Only
    ctx=256 is 3/3."""
    vl = load_val_loss()
    print("\n=== Point 1b: per-seed sign counts, normalized metric, S vs D1 ===")
    for c in CTXS:
        signs = []
        for s in SEEDS:
            sn = PPL["S"][s][c] / math.exp(vl["S"][s])
            dn = PPL["D1"][s][c] / math.exp(vl["D1"][s])
            signs.append("S" if sn < dn else "D1")
        print(f"  ctx{c}: {signs}  S wins {signs.count('S')}/3")


def check_2b_q0_per_seed():
    """Follow-up to point 2: is the Q0 rank climb seed-unanimous at the two
    load-bearing comparisons, or just a pooled-mean artifact?"""
    print("\n=== Point 2b: Q0 per-seed, does it beat S at 2048 / D1,D0p at 4096? ===")
    for s in SEEDS:
        q, sv = PPL["Q0"][s][2048], PPL["S"][s][2048]
        print(f"  ctx2048 seed{s}: Q0={q} S={sv}  "
              f"{'Q0 beats S' if q < sv else 'S beats Q0'}")
    for s in SEEDS:
        q, d1v, d0pv = PPL["Q0"][s][4096], PPL["D1"][s][4096], PPL["D0p"][s][4096]
        print(f"  ctx4096 seed{s}: Q0={q} D1={d1v} D0p={d0pv}  "
              f"Q0<D1={q < d1v}  Q0<D0p={q < d0pv}")


def check_3_batch_pairing():
    """Review point 3 (refuted): does batch_size shape change which
    windows torch.randint draws from a fixed-seed CPU generator? No GPU
    needed -- ValOnlyDataset.get_batch's index draw is CPU-side
    regardless of eval device."""
    import torch

    def draw(seed, batch_size, iters, n):
        g = torch.Generator().manual_seed(seed)
        out = []
        for _ in range(iters):
            out.extend(torch.randint(n, (batch_size,), generator=g).tolist())
        return out

    n = 4884400 - 4096 - 1
    seed = 1337 * 1000 + 4096
    a = draw(seed, 8, 32, n)     # dense variants' config at ctx=4096
    b = draw(seed, 2, 128, n)    # S/X's config at ctx=4096
    identical = a == b
    print(f"\n=== Point 3: batch-size-dependent draw sequence? "
          f"identical={identical} ===")
    assert identical, "REGRESSION: batch size now changes the draw sequence"


if __name__ == "__main__":
    check_1_normalized_crossing()
    check_1b_normalized_per_seed()
    check_2_q0_rank()
    check_2b_q0_per_seed()
    check_3_batch_pairing()
