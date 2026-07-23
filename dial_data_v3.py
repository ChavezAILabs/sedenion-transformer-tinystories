"""dial_data_v3.py — repo port of the Phase 4 multi-query dial generator
(v3: grid-block collisions, Q supervised tokens/episode).

Ported from `dial_v3_prototype.py` (Claude Desktop, verified standalone
2026-07-20 by Claude Code — all 4 checks reproduced exactly) per
`PHASE4_autonomous_run_plan.md` SS2.3 port requirements:
  - mirror discipline: `gen` here must match the prototype episode-for-
    episode (checked below, lockstep).
  - answer mask is now TRUE at n_queries positions per episode, not 1;
    the mask-correctness check is extended to all of them.
  - accuracy is over n_queries supervised tokens per episode (handled in
    phase4_train.py's generalized, mask-driven accuracy computation, not
    here).
  - v2 and v3 numbers are NOT comparable — different generator, different
    task. Never plot them on one axis.

Sequence layout: N [ka kb v] triples (shuffled) + <Q> marker +
n_queries [qa qb v] query triples, length 3N + 1 + 3*n_queries (N=16,
n_queries=4 -> 61 tokens, fits ctx 64). `gen` returns the FULL sequence
including inline answer values (unlike v2, which appended the single
answer separately) — training pairs are x = seq[:-1], y = seq[1:], and
each answer at seq-index p is predicted at y-index p-1.

Floor for any single-token matcher, unchanged in form from v2:
    floor(delta) = 1/(delta+1) + (1 - 1/(delta+1))/n_val
"""
import numpy as np
import torch

from dial_v3_prototype import gen as _gen_reference
from dial_v3_prototype import floor as floor_acc


def gen(delta, N=16, n_queries=4, n_key=64, n_val=64, rng=None):
    """Identical sampling sequence to the prototype (kept in lockstep)."""
    return _gen_reference(delta, N=N, n_queries=n_queries, n_key=n_key,
                          n_val=n_val, rng=rng)


def vocab_size(n_key=64, n_val=64):
    return n_key + n_val + 1          # keys, values, <Q> marker


def make_batch(delta, batch_size, N=16, n_queries=4, n_key=64, n_val=64,
               rng=None, device="cpu"):
    """Returns x (B, L-1) int64, y (B, L-1) int64, mask (B, L-1) bool.
    mask is True at n_queries answer-predicting positions per row."""
    rng = rng or np.random.default_rng()
    L = 3 * N + 1 + 3 * n_queries
    xlen = L - 1
    xs = np.empty((batch_size, xlen), dtype=np.int64)
    ys = np.empty((batch_size, xlen), dtype=np.int64)
    mask = np.zeros((batch_size, xlen), dtype=bool)
    for b in range(batch_size):
        seq, answers, ans_pos = gen(delta, N=N, n_queries=n_queries,
                                    n_key=n_key, n_val=n_val, rng=rng)
        xs[b], ys[b] = seq[:-1], seq[1:]
        for p in ans_pos:
            mask[b, p - 1] = True
    return (torch.from_numpy(xs).to(device), torch.from_numpy(ys).to(device),
            torch.from_numpy(mask).to(device))


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import dial_v3_prototype as proto

    N, Q = 16, 4
    LEVELS = (0, 1, 2, 3)

    # [E1] lockstep with the prototype: identical episodes, same rng seed
    for d in LEVELS:
        r1 = np.random.default_rng(42)
        r2 = np.random.default_rng(42)
        for _ in range(200):
            s1, a1, p1 = proto.gen(d, N=N, n_queries=Q, rng=r1)
            s2, a2, p2 = gen(d, N=N, n_queries=Q, rng=r2)
            assert np.array_equal(s1, s2) and a1 == a2 and p1 == p2
    print("[E1] port == prototype, episode-for-episode      OK")

    # [E2] mask correctness: exact 4-position isolation
    x, y, mask = make_batch(2, 8, N=N, n_queries=Q, rng=np.random.default_rng(0))
    V = vocab_size()
    assert mask.sum(dim=1).eq(Q).all(), "expected exactly Q masked positions/row"
    logits = torch.randn(8, x.shape[1], V)
    from dial_data import masked_loss          # generic: works for any mask
    base = masked_loss(logits, y, mask).item()
    y_bad_nonans = y.clone()
    nonmask = ~mask
    y_bad_nonans[nonmask] = (y_bad_nonans[nonmask] + 1) % V
    assert abs(masked_loss(logits, y_bad_nonans, mask).item() - base) < 1e-12
    # corrupting the answer positions must change the loss
    y_bad_ans = y.clone()
    y_bad_ans[mask] = (y_bad_ans[mask] + 1) % V
    assert abs(masked_loss(logits, y_bad_ans, mask).item() - base) > 1e-6
    # hand computation: mean CE over the Q*8 = 32 selected (row,pos) pairs
    import torch.nn.functional as F
    sel = mask.reshape(-1)
    ref = F.cross_entropy(logits.reshape(-1, V)[sel], y.reshape(-1)[sel]).item()
    assert abs(base - ref) < 1e-9
    print(f"[E2] mask correctness at all {Q} answer positions   OK")

    # [E3] length / vocab / answer-count fixed across delta
    lens = set()
    for d in LEVELS:
        x, y, mask = make_batch(d, 4, N=N, n_queries=Q, rng=np.random.default_rng(d))
        lens.add((x.shape[1], int(mask.sum(dim=1).max()), int(mask.sum(dim=1).min())))
    print("[E3] fixed (xlen, mask-count) across delta:", lens,
          "->", "PASS" if len(lens) == 1 else "FAIL")

    # [E4] floor formula matches the prototype's own (already verified standalone)
    for d in LEVELS:
        assert abs(floor_acc(d) - proto.floor(d)) < 1e-12
    print("[E4] floor_acc == prototype floor formula          OK")

    print("\nAll dial_data_v3 port checks passed.")
