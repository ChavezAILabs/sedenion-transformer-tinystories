"""dial_data.py — repo port of the Phase 4 selective-recall dial task
(v2, balanced collisions: delta prefix-collisions AND delta suffix-
collisions, constraint 2*delta+1 <= N; the v1 prefix-only design was
leaky to a suffix-only matcher — design doc SS1 v2 correction; v1
archived as dial_selective_recall_prototype_v1_LEAKY.py).

Mirror discipline: `dial_selective_recall_prototype.py` (v2) is the
reference implementation; `gen` here must produce IDENTICAL episodes
given the same rng (tested below, episode-for-episode), and design-doc
SS2 checks 1-4 must reproduce INCLUDING the full single-token heuristic
suite (prefix, suffix, either) — the standing mechanism for testing any
newly proposed shortcut before calibration results are trusted.

Training layout per episode (answer appended for LM training):
    tokens = [ka kb v]*N + [<Q> qa qb ans]        (len 3N+4)
    x = tokens[:-1], y = tokens[1:]               (len 3N+3)
    loss mask: True ONLY at the final position (qb -> ans prediction).
Graded metric = mean loss at masked positions (design doc SS3).

Non-binding floor, scoped to "best verified single-token strategy":
floor(delta) = 1/(delta+1) + (1 - 1/(delta+1))/n_val — values drawn WITH
replacement (prototype semantics kept; without-replacement is a v1.0
decision). In v2 this floor holds for BOTH prefix and suffix matchers.

Seed discipline (design doc SS5.2): all episode randomness flows from the
numpy Generator passed in; the trainer derives it from (data_seed) alone
so every variant at a seed sees identical episodes; eval episodes are
drawn from default_rng((data_seed, step, delta)).
"""
import numpy as np
import torch

from dial_selective_recall_prototype import gen as _gen_reference


def gen(delta, N=16, n_key=64, n_val=64, rng=None):
    """Identical sampling sequence to the prototype (kept in lockstep)."""
    return _gen_reference(delta, N=N, n_key=n_key, n_val=n_val, rng=rng)


def vocab_size(n_key=64, n_val=64):
    return n_key + n_val + 1          # keys, values, <Q> marker


def make_batch(delta, batch_size, N=16, n_key=64, n_val=64, rng=None,
               device="cpu"):
    """Returns x (B, 3N+3) int64, y (B, 3N+3) int64, mask (B, 3N+3) bool.
    mask is True only at the answer-predicting position."""
    rng = rng or np.random.default_rng()
    L = 3 * N + 3
    xs = np.empty((batch_size, L), dtype=np.int64)
    ys = np.empty((batch_size, L), dtype=np.int64)
    for b in range(batch_size):
        seq, ans = gen(delta, N=N, n_key=n_key, n_val=n_val, rng=rng)
        toks = np.concatenate([seq, [ans]])
        xs[b], ys[b] = toks[:-1], toks[1:]
    mask = np.zeros((batch_size, L), dtype=bool)
    mask[:, -1] = True
    return (torch.from_numpy(xs).to(device), torch.from_numpy(ys).to(device),
            torch.from_numpy(mask).to(device))


def masked_loss(logits, y, mask):
    """Mean cross-entropy over masked positions only."""
    import torch.nn.functional as F
    sel = mask.reshape(-1)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1))[sel],
                           y.reshape(-1)[sel])


def floor_acc(delta, n_val=64):
    p = 1.0 / (delta + 1)
    return p + (1 - p) / n_val


# ----------------------------------------------------------------------
if __name__ == "__main__":
    from collections import Counter
    from dial_selective_recall_prototype import (
        oracle, prefix_heuristic, suffix_heuristic, either_heuristic)

    LEVELS = (0, 2, 4, 7)          # 2*delta+1 <= N=16 caps delta at 7

    # [D1] lockstep with the prototype: identical episodes, same rng seed
    for d in LEVELS:
        r1 = np.random.default_rng(42)
        r2 = np.random.default_rng(42)
        for _ in range(200):
            s1, a1 = _gen_reference(d, rng=r1)
            s2, a2 = gen(d, rng=r2)
            assert np.array_equal(s1, s2) and a1 == a2
    print("[D1] port == prototype, episode-for-episode      OK")

    # [D2] strengthened check 1: length, token range, marker position,
    # answer in value range — 3000 episodes per delta
    for d in LEVELS:
        r = np.random.default_rng(d)
        for _ in range(3000):
            s, a = gen(d, rng=r)
            assert len(s) == 51 and s.min() >= 0 and s.max() <= 128
            assert s[-3] == 128 and 64 <= a < 128
    print("[D2] length/range/marker/answer over 3k eps x4   OK")

    # [D3] design check 2: unigrams near-matched across the dial
    def unigram(d, n=3000):
        r = np.random.default_rng(d + 100)
        c = Counter()
        for _ in range(n):
            s, _ = gen(d, rng=r)
            c.update(s.tolist())
        p = np.array([c.get(i, 0) for i in range(129)], float) + 0.5
        return p / p.sum()
    P = {d: unigram(d) for d in LEVELS}
    kl = lambda p, q: float(np.sum(p * np.log(p / q)))
    kls = [kl(P[0], P[d]) for d in LEVELS[1:]]
    assert max(kls) < 0.01
    print(f"[D3] unigram KL {['%.5f' % k for k in kls]} (<0.01)  OK")

    # [D4] design checks 3-4: oracle exact, FULL single-token heuristic
    # suite at/below floor (prefix AND suffix track floor; either below)
    from dial_selective_recall_prototype import floor as proto_floor
    for d in LEVELS:
        r = np.random.default_rng(7)
        n = 4000
        o = p = s_ = e = 0
        for _ in range(n):
            sq, a = gen(d, rng=r)
            o += oracle(sq) == a
            p += prefix_heuristic(sq) == a
            s_ += suffix_heuristic(sq) == a
            e += either_heuristic(sq) == a
        fl = proto_floor(d)
        assert o == n, f"oracle broke at delta={d}"
        assert abs(proto_floor(d) - floor_acc(d)) < 1e-12
        if d > 0:
            assert abs(p / n - fl) < 0.05 and abs(s_ / n - fl) < 0.05, \
                (d, p / n, s_ / n, fl)
            assert e / n < fl + 0.05
        print(f"[D4] d={d}: oracle 1.000  prefix {p/n:.3f}  suffix {s_/n:.3f}"
              f"  either {e/n:.3f}  floor {fl:.3f}   OK")

    # [D5] mask correctness: exact-position isolation
    x, y, mask = make_batch(4, 8, rng=np.random.default_rng(0))
    V = vocab_size()
    logits = torch.randn(8, x.shape[1], V)
    base = masked_loss(logits, y, mask).item()
    y2 = y.clone(); y2[:, :-1] = (y2[:, :-1] + 1) % V     # corrupt non-answer
    assert abs(masked_loss(logits, y2, mask).item() - base) < 1e-12
    y3 = y.clone(); y3[:, -1] = (y3[:, -1] + 1) % V       # corrupt answer
    assert abs(masked_loss(logits, y3, mask).item() - base) > 1e-6
    # and against a hand computation
    import torch.nn.functional as F
    ref = F.cross_entropy(logits[:, -1, :], y[:, -1]).item()
    assert abs(base - ref) < 1e-9
    print("[D5] answer-position loss mask exact              OK")

    # [D6] floor formula vs 20k-episode simulation, BOTH matchers
    for d in (4, 7):
        r = np.random.default_rng(11)
        n, hp, hs = 20000, 0, 0
        for _ in range(n):
            s, a = gen(d, rng=r)
            hp += prefix_heuristic(s) == a
            hs += suffix_heuristic(s) == a
        f = floor_acc(d)
        assert abs(hp / n - f) < 0.01, ("prefix", d, hp / n, f)
        assert abs(hs / n - f) < 0.01, ("suffix", d, hs / n, f)
    print("[D6] floor formula matches simulation (prefix+suffix) OK")

    print("\nAll dial_data checks passed.")
