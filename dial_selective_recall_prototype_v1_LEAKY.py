"""Prototype generator for the Phase 4 selective-recall dial task.

Design doc: PHASE4_dial_task_design.md. Verified properties (checks 1-4):
fixed length/vocab across delta, near-matched unigrams, oracle=1.0 at all
delta, prefix-only heuristic ~ 1/(delta+1). Run this file to re-verify.
NumPy-mirror-first: the repo port must reproduce these checks exactly.
"""
import numpy as np
from collections import Counter


def gen(delta, N=16, n_key=64, n_val=64, rng=None):
    """One episode. Returns (token sequence, answer token).

    Layout: N shuffled [ka kb v] triples, then [<Q> qa qb].
    Target pair is (qa, qb) -> answer. delta distractor pairs share qa
    (near-collisions); the rest have distinct first key tokens.
    Token spaces: keys [0, n_key), values [n_key, n_key+n_val),
    query marker = n_key + n_val.
    """
    assert 0 <= delta <= N - 1
    rng = rng or np.random.default_rng()
    Qtok = n_key + n_val
    qa, qb = rng.integers(0, n_key, 2)
    keys = [(qa, qb)]
    used_b = {qb}
    for _ in range(delta):
        b = rng.integers(0, n_key)
        while b in used_b:
            b = rng.integers(0, n_key)
        used_b.add(b)
        keys.append((qa, b))
    used_a = {qa}
    for _ in range(N - 1 - delta):
        a = rng.integers(0, n_key)
        while a in used_a:
            a = rng.integers(0, n_key)
        used_a.add(a)
        keys.append((a, rng.integers(0, n_key)))
    vals = rng.integers(n_key, n_key + n_val, N)
    order = rng.permutation(N)
    seq, ans = [], None
    for idx in order:
        ka, kb = keys[idx]
        seq += [ka, kb, vals[idx]]
        if idx == 0:
            ans = vals[idx]
    seq += [Qtok, qa, qb]
    return np.array(seq), ans


def oracle(seq):
    qa, qb = seq[-2], seq[-1]
    for i in range(0, len(seq) - 3, 3):
        if seq[i] == qa and seq[i + 1] == qb:
            return seq[i + 2]


def prefix_heuristic(seq):
    qa = seq[-2]
    for i in range(0, len(seq) - 3, 3):
        if seq[i] == qa:
            return seq[i + 2]


def _unigram(delta, n=3000, dim=129):
    r = np.random.default_rng(delta + 100)
    c = Counter()
    for _ in range(n):
        s, _ = gen(delta, rng=r)
        c.update(s.tolist())
    p = np.array([c.get(i, 0) for i in range(dim)], float) + 0.5
    return p / p.sum()


def _kl(p, q):
    return float(np.sum(p * np.log(p / q)))


if __name__ == "__main__":
    levels = [0, 4, 12]
    lens = set()
    for d in levels:
        s, _ = gen(d, rng=np.random.default_rng(1))
        lens.add((len(s), int(s.max())))
    print("check 1 (fixed length/vocab):", lens, "->", "PASS" if len(lens) == 1 else "FAIL")
    P = {d: _unigram(d) for d in levels}
    k4, k12 = _kl(P[0], P[4]), _kl(P[0], P[12])
    print(f"check 2 (unigram KL vs delta=0): {k4:.5f}, {k12:.5f} ->",
          "PASS" if max(k4, k12) < 0.01 else "FAIL")
    for d in levels:
        r = np.random.default_rng(7)
        n, ok_o, ok_p = 2000, 0, 0
        for _ in range(n):
            s, a = gen(d, rng=r)
            ok_o += oracle(s) == a
            ok_p += prefix_heuristic(s) == a
        print(f"check 3/4 delta={d:2d}: oracle={ok_o/n:.3f} (expect 1.0)  "
              f"prefix-only={ok_p/n:.3f} (expect ~{1/(d+1):.3f})")
