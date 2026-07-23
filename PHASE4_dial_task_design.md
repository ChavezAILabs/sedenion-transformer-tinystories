# Phase 4 dial task family — design (2026-07-19)

Fills blocking item §9.5 of `ZDA_phase4_spec.md`. Decision (project owner,
2026-07-19): the registered dial is the mechanism's best case — selective
recall with distractor density — with k-hop variable binding designated as
the pre-registered follow-up family if H4d passes. The alignment risk of a
mechanism-friendly task is accepted and stated openly: H4b′'s controls (R,
X) run the identical task, so a task-shaped artifact that isn't
frame-specific cannot survive attribution.

## 1. Task: near-collision selective recall

Each episode is a sequence of N key–value pairs followed by a query:

    [k1a k1b v1] [k2a k2b v2] … [kNa kNb vN] <Q> qa qb → answer = v of (qa,qb)

Keys are 2-token; values 1-token; token spaces for keys, values, and the
query marker are disjoint. One pair is the target. **δ of the N−1
distractor pairs are near-collisions: they share the first key token qa
with the target and differ only in the second.** The remaining N−1−δ
distractors have distinct first tokens. Pair order is uniformly shuffled.

**v2 CORRECTION (2026-07-19, same day — before any calibration
conclusions were drawn):** the v1 design above, with prefix-only
collisions, was leaky. Because near-collisions were guaranteed to differ
in the *second* key token, a suffix-only matcher scored 0.89 at δ=0 and
**rose to 0.97 at δ=12** — turning the dial up removed suffix-competitors
and made the single-token shortcut easier. The v1 "provably targets
binding" claim was an overclaim: check 4 tested one non-binding heuristic
and the conclusion was stated as if it covered all of them. Error was in
the task design (analysis side), caught while deriving why the corrected
floor formula is a floor.

**v2 design: balanced collisions.** δ prefix-collisions (share qa, differ
in kb) AND δ suffix-collisions (share qb, differ in ka), constraint
2δ+1 ≤ N. No single key token identifies the target; every verified
single-token heuristic (prefix, suffix, either-token) degrades toward
floor(δ) = 1/(δ+1) + (1−1/(δ+1))/n_val — the coincidence-corrected
formula, now valid for both tokens — while the full-binding oracle stays
at 1.000. At δ = 0 single-token matching suffices (checkers); at high δ
only joint two-token binding with active ignoring of 2δ near-matches
works (chess). Scope of the floor claim, stated precisely this time:
"best *verified single-token* strategy." Analysis suggests single-token
exhausts the non-binding paths (values i.i.d., pair order uniform, no
positional or frequency route that doesn't reduce to the same 1/(δ+1)
bound), but that is argument, not proof — any newly proposed shortcut
gets tested against the generator before calibration results are
trusted, and the shortcut-check suite in the prototype is the standing
mechanism for that.

Why near-collision density rather than pair count or noise fraction: it
is the only dial found that moves *binding demand* while holding sequence
length, pair count, vocabulary, and (nearly) unigram statistics exactly
fixed across δ. This is selective ignoring in the literal sense the K3
mechanism story requires, which is precisely why it is the best-case
dial.

## 2. Prototype verification — v2 (NumPy, toy config N=16, 64 keys, 64 values)

All checks run on the v2 generator, 3000–4000 episodes per cell, levels
δ ∈ {0, 2, 4, 7} (N = 16 caps δ at 7 via 2δ+1 ≤ N):

1. **Length and vocabulary exactly constant across δ** (51 tokens, same
   token range at every level). VERIFIED.
2. **Unigram statistics near-matched:** KL vs δ=0 of 0.0008 / 0.0012 /
   0.0015 at δ = 2/4/7. The residual trace is now symmetric — qa and qb
   each appear δ extra times. A repetition-detector exploiting it still
   faces δ+1 candidates per token, reducing to the same floor. All K3
   variants see the identical trace. VERIFIED.
3. **Well-posed at every δ:** the full-binding oracle scores 1.000 at
   all levels. VERIFIED.
4. **The dial dials binding demand — against the full single-token
   suite:** prefix-only 1.000 → 0.140, suffix-only 0.894 → 0.145,
   either-token 0.894 → 0.084 across δ = 0 → 7, with prefix and suffix
   tracking floor(δ) = 1/(δ+1) + (1−1/(δ+1))/64 (1.000 / 0.344 / 0.213 /
   0.139). The v1 leak (suffix rising with δ) is closed. VERIFIED, with
   the claim scoped to the verified heuristic suite per §1.

Generator prototype: `dial_selective_recall_prototype.py` (v2; v1
archived as `dial_selective_recall_prototype_v1_LEAKY.py` for the
record). NumPy-mirror-first applies: the repo port must reproduce checks
1–4 including the full heuristic suite before calibration is trusted.

## 3. Registered configuration (values TBD at calibration, form fixed now)

- Episode parameters N, key/value vocab sizes, and ctx packing (episodes
  per training sequence) fixed at calibration; identical across δ and
  across all variants; same tokenizer/vocab object for every run.
- δ levels: 3–4, including δ = 0 (the checkers anchor of the dial) and a
  top level at which D0p shows clear headroom (spec §8.5: not saturated,
  not floored — calibration runs are ungraded). Level choice guided by
  the 1/(δ+1) scale, frozen at v1.0.
- **Graded metric: mean loss at the answer position** (continuous,
  compatible with the 2× pooled-std machinery); answer-token accuracy
  reported descriptively. Each δ level is a separate task condition with
  its own runs — no mixed-δ training in the graded grid (a mixed-δ
  curriculum is a different experiment; door closed here to prevent
  drift).
- H4d reduction set per spec §3 (v0.3 — variant R dropped as vacuous,
  memo §8 third closed door): {S, X, D0p} at each dial level; full grid
  on the anchor task only.

## 4. Designated follow-up: k-hop variable binding (outline, frozen in form)

Triggered only by H4d passing. Chains of assignments (x1 := v; x2 := x1;
… ; xk := x(k−1); query xk) with distractor chains; δ = k. The hard-case
test: compositional depth rather than filtering. Full generator design,
matched-statistics analysis, and its own calibration get a v1.0 process
of their own at trigger time; registering the family and dial parameter
now prevents an improvised hard test after a friendly-task win. If H4d
fails, this family is not run and the negative stands on the registered
dial.

## 5. Relay list (Claude Code)

1. **Re-port to v2 (balanced collisions)**; reproduce §2 checks 1–4
   including the full single-token heuristic suite. The running D0p
   sweep uses the leaky v1 generator: its results are void for δ-level
   selection. Falsifiable prediction, logged before the sweep reads out:
   if the leak diagnosis is right, v1-sweep D0p accuracy at high δ
   should converge toward the suffix band (~0.92–0.97), not the stated
   floor (~0.09–0.21) and not full ceiling — the sweep retroactively
   becomes an empirical shortcut-detection test.
2. **N decision at calibration:** 2δ+1 ≤ N caps the dial. Either keep
   N = 16 with registered levels drawn from {0, 2, 4, 7}, or raise N
   (e.g. N = 26 for δ up to 12) — N is frozen and identical across all
   δ and variants either way.
3. Seed discipline: episode sampling must follow the Phase 3 invariant —
   data order seeded independently of model RNG, identical episodes
   across variants at a seed, eval sets seeded by (seed, step, δ).
4. Calibration protocol per spec §8.5 once the v2 port is green: pick δ
   levels off the D0p headroom curve; ungraded.
5. Answer-position loss requires a loss mask in the training/eval loop —
   flag as a harness change needing its own test (mask correctness:
   loss over non-answer positions must be excluded exactly).
