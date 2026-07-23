"""Prototype generator for the Phase 4 selective-recall dial task — v2.

v2 (2026-07-19): BALANCED COLLISIONS. v1 was leaky: near-collisions shared
only the target's prefix token, so a suffix-only matcher scored 0.89->0.97
*rising* with delta — the dial made the single-token shortcut easier. v2
plants delta prefix-collisions AND delta suffix-collisions, so no single
key token identifies the target; every verified single-token heuristic
(prefix, suffix, either) degrades toward floor(delta) = 1/(delta+1) +
(1-1/(delta+1))/n_val while the full-binding oracle stays at 1.0.
Constraint: 2*delta + 1 <= N. Scope of the floor claim: "best verified
single-token strategy" — analysis suggests single-token exhausts
non-binding paths (values iid, order uniform), but that is argument, not
proof; new shortcut candidates should be tested here before trusting
calibration. Run this file to re-verify all checks.
"""
import numpy as np
from collections import Counter


def gen(delta, N=16, n_key=64, n_val=64, rng=None):
    """One episode. delta prefix-collisions (share qa, differ in kb) and
    delta suffix-collisions (share qb, differ in ka); remaining distractors
    have distinct first tokens and random second tokens."""
    assert 2 * delta + 1 <= N, "need 2*delta+1 <= N"
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
    for _ in range(delta):
        a = rng.integers(0, n_key)
        while a in used_a:
            a = rng.integers(0, n_key)
        used_a.add(a)
        keys.append((a, qb))
    for _ in range(N - 1 - 2 * delta):
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


def suffix_heuristic(seq):
    qb = seq[-1]
    for i in range(0, len(seq) - 3, 3):
        if seq[i + 1] == qb:
            return seq[i + 2]


def either_heuristic(seq):
    qa, qb = seq[-2], seq[-1]
    for i in range(0, len(seq) - 3, 3):
        if seq[i] == qa or seq[i + 1] == qb:
            return seq[i + 2]


def floor(delta, n_val=64):
    p = 1.0 / (delta + 1)
    return p + (1 - p) / n_val


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
    levels = [0, 2, 4, 7]
    lens = set()
    for d in levels:
        s, _ = gen(d, rng=np.random.default_rng(1))
        lens.add((len(s), int(s.max())))
    print("check 1 (fixed length/vocab):", lens, "->", "PASS" if len(lens) == 1 else "FAIL")
    P = {d: _unigram(d) for d in levels}
    kls = [_kl(P[0], P[d]) for d in levels[1:]]
    print("check 2 (unigram KL vs delta=0):", ["%.5f" % k for k in kls], "->",
          "PASS" if max(kls) < 0.01 else "FAIL")
    print("checks 3/4 (oracle vs ALL single-token heuristics):")
    all_pass = True
    for d in levels:
        r = np.random.default_rng(7)
        n = 4000
        o = p = s_ = e = 0
        for _ in range(n):
            sq, a = gen(d, rng=r)
            o += oracle(sq) == a
            p += prefix_heuristic(sq) == a
            s_ += suffix_heuristic(sq) == a
            e += either_heuristic(sq) == a
        fl = floor(d)
        near = all(abs(x / n - fl) < 0.05 for x in (p, s_)) if d > 0 else True
        all_pass &= (o == n) and near
        print(f"  delta={d}: oracle={o/n:.3f} prefix={p/n:.3f} suffix={s_/n:.3f} "
              f"either={e/n:.3f} floor={fl:.3f}")
    print("->", "PASS" if all_pass else "FAIL")
