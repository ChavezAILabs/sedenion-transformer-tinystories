"""Phase 4 dial task generator — v3 (multi-query, grid-block collisions).

v3 changes from v2 (2026-07-20):
  * MULTI-QUERY. v2 supplied one supervised token per 51-token episode.
    MQAR (Arora et al. 2023) is defined by multiple recalls per sequence;
    v2 was the pre-MQAR single-query formulation Zoology moved away from.
  * GRID-BLOCK COLLISIONS. v2 planted collisions per-query, so Q queries
    needed Q*(2d+1) <= N and the dial collapsed under multi-query.
    v3 arranges keys in m x m blocks (m = delta+1): every key in a block
    shares its prefix with the m-1 others in its row and its suffix with
    the m-1 in its column. Collision structure is a property of the pair
    set, so ANY block key is a valid query and Q is decoupled from delta.
    Constraint becomes B*m^2 <= N with B = N // m^2 blocks.
  * DISJOINT TOKEN POOLS. Blocks and generic distractors draw distinct key
    tokens, so the shortcut floor is exact rather than approximate.

Floor for any single-token matcher (prefix-only or suffix-only), unchanged
in form from v2:  floor(delta) = 1/(delta+1) + (1 - 1/(delta+1))/n_val
Either-token matcher is strictly lower: 1/(2m-1) + coincidence.

Layout: N [ka kb v] triples (shuffled), <Q> marker, then Q [qa qb v] query
triples. Supervised positions are the query value slots.
Length = 3N + 1 + 3Q  (N=16, Q=4 -> 61 tokens, fits ctx 64).
"""
import numpy as np
from collections import Counter


def gen(delta, N=16, n_queries=4, n_key=64, n_val=64, rng=None):
    """One episode. Returns (seq, answers, answer_positions)."""
    m = delta + 1
    B = N // (m * m)
    assert B >= 1, f"delta={delta} needs m^2={m*m} <= N={N}"
    assert n_queries <= B * m * m, "not enough block keys to query"
    rng = rng or np.random.default_rng()
    Qtok = n_key + n_val
    n_generic = N - B * m * m

    n_pref = B * m + n_generic
    n_suff = B * m + n_generic
    pref = rng.choice(n_key, size=n_pref, replace=False)
    suff = rng.choice(n_key, size=n_suff, replace=False)

    keys = []
    for b in range(B):                      # grid blocks
        pa = pref[b * m:(b + 1) * m]
        pb = suff[b * m:(b + 1) * m]
        for i in range(m):
            for j in range(m):
                keys.append((int(pa[i]), int(pb[j])))
    n_block = len(keys)
    for g in range(n_generic):              # generic distractors
        keys.append((int(pref[B * m + g]), int(suff[B * m + g])))

    vals = rng.integers(n_key, n_key + n_val, N)
    order = rng.permutation(N)
    q_idx = rng.choice(n_block, size=n_queries, replace=False)

    seq, answers, ans_pos = [], [], []
    for idx in order:
        ka, kb = keys[idx]
        seq += [ka, kb, int(vals[idx])]
    seq.append(Qtok)
    for qi in q_idx:
        ka, kb = keys[qi]
        seq += [ka, kb]
        ans_pos.append(len(seq))
        seq.append(int(vals[qi]))
        answers.append(int(vals[qi]))
    return np.array(seq), answers, ans_pos


def _pair_region(seq, N):
    return [(seq[i], seq[i + 1], seq[i + 2]) for i in range(0, 3 * N, 3)]


def _queries(seq, N, n_queries):
    base = 3 * N + 1
    return [(seq[base + 3 * t], seq[base + 3 * t + 1]) for t in range(n_queries)]


def oracle(seq, N, n_queries):
    out = []
    pairs = _pair_region(seq, N)
    for qa, qb in _queries(seq, N, n_queries):
        out.append(next((v for a, b, v in pairs if a == qa and b == qb), None))
    return out


def h_prefix(seq, N, n_queries):
    pairs = _pair_region(seq, N)
    return [next((v for a, b, v in pairs if a == qa), None)
            for qa, qb in _queries(seq, N, n_queries)]


def h_suffix(seq, N, n_queries):
    pairs = _pair_region(seq, N)
    return [next((v for a, b, v in pairs if b == qb), None)
            for qa, qb in _queries(seq, N, n_queries)]


def h_either(seq, N, n_queries):
    pairs = _pair_region(seq, N)
    return [next((v for a, b, v in pairs if a == qa or b == qb), None)
            for qa, qb in _queries(seq, N, n_queries)]


def floor(delta, n_val=64):
    p = 1.0 / (delta + 1)
    return p + (1 - p) / n_val


def floor_either(delta, n_val=64):
    m = delta + 1
    p = 1.0 / (2 * m - 1)
    return p + (1 - p) / n_val


def _unigram(delta, N, Q, n=2000, dim=129):
    r = np.random.default_rng(delta + 500)
    c = Counter()
    for _ in range(n):
        s, _, _ = gen(delta, N=N, n_queries=Q, rng=r)
        c.update(s.tolist())
    p = np.array([c.get(i, 0) for i in range(dim)], float) + 0.5
    return p / p.sum()


def _kl(p, q):
    return float(np.sum(p * np.log(p / q)))


if __name__ == "__main__":
    N, Q = 16, 4
    levels = [0, 1, 2, 3]
    print(f"=== v3 dial: N={N}, queries={Q}, levels={levels} ===\n")

    lens = set()
    for d in levels:
        s, a, ap = gen(d, N=N, n_queries=Q, rng=np.random.default_rng(1))
        lens.add((len(s), int(s.max()), len(a)))
    print("check 1 (fixed length / vocab / #answers):", lens,
          "->", "PASS" if len(lens) == 1 else "FAIL")

    P = {d: _unigram(d, N, Q) for d in levels}
    kls = [_kl(P[0], P[d]) for d in levels[1:]]
    print("check 2 (unigram KL vs delta=0):", ["%.5f" % k for k in kls],
          "->", "PASS" if max(kls) < 0.01 else "FAIL")

    print("\ncheck 3/4 (oracle vs full single-token suite, 3000 episodes):")
    ok = True
    for d in levels:
        r = np.random.default_rng(11)
        n = 3000
        tot = o = p = s_ = e = 0
        for _ in range(n):
            sq, ans, _ = gen(d, N=N, n_queries=Q, rng=r)
            po, pp, ps, pe = (oracle(sq, N, Q), h_prefix(sq, N, Q),
                              h_suffix(sq, N, Q), h_either(sq, N, Q))
            for t in range(Q):
                tot += 1
                o += po[t] == ans[t]
                p += pp[t] == ans[t]
                s_ += ps[t] == ans[t]
                e += pe[t] == ans[t]
        fl, fe = floor(d), floor_either(d)
        good = (o == tot) and abs(p / tot - fl) < 0.03 and abs(s_ / tot - fl) < 0.03
        ok &= good
        print(f"  delta={d}: oracle={o/tot:.3f}  prefix={p/tot:.3f}  "
              f"suffix={s_/tot:.3f}  either={e/tot:.3f}   "
              f"| floor={fl:.3f} floor_either={fe:.3f}  {'ok' if good else 'CHECK'}")
    print("->", "PASS" if ok else "FAIL")

    print("\nsupervised tokens per episode: v2=1, v3=%d  (%.0fx density)" % (Q, Q))
    s, a, ap = gen(2, N=N, n_queries=Q, rng=np.random.default_rng(3))
    print("example seq len:", len(s), " answer positions:", ap)
