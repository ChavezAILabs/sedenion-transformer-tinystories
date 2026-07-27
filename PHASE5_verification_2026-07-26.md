# PHASE 5 verification note — 2026-07-26

**Status:** Claude Code repo-side verification of claims proposed in
`PHASE5_PLAN.md` §2 and §10, and of a citation surfaced afterward that
changes what those claims are evidence *for*. All numbers below were
computed fresh against this repo's own code (`sedenion_kernel.py`,
`phase4_layers.py`) — nothing here is transcribed from a chat-side session
without an independent rerun.

**Convention:** Baez throughout, matching `sedenion_kernel.cd_mult`.

**Supersedes:** an earlier draft of this note that framed bilaterality as
new. It is not — see §2.

---

## 1. Enumeration — the 84/336 counting lattice

Ran `sedenion_kernel.py`'s own exhaustive two-blade search fresh and cross-
tabulated the result:

| object | count |
|---|---|
| distinct index sets {a,b} | 42 |
| two-blade elements (low index sign fixed +1) | 84 |
| partner multiplicity per element | 4, uniformly (no exceptions) |
| ordered pairs (x,y) with x⊛y=0 | 336 |
| self-paired instances (x=y) | 0 |
| unordered pairs {x,y} | 168 |

All confirmed by direct computation on `sedenion_kernel.structure_tensor(16)`
(`336/2 == 168` exactly, with zero self-paired instances — a real check, not
an assumed symmetry, since a self-paired instance would break it).

All 84 two-blade elements match Reggiani's (arXiv:2411.18881) stated index
rule exactly: i ∈ {1..7}, j ∈ {9..15}, i ≠ j−8 (7×7 − 7 = 42 index sets ×
2 signs = 84). Three independent routes now converge on the same 84/336
structure: this repo's exhaustive search, Reggiani's published index rule,
and (per a parallel session's independent from-scratch Cayley-Dickson
construction, not re-verified line-by-line here but consistent with the
above) a from-scratch enumeration.

**Note the 168 collision.** Two unrelated objects both have count 168:
"two-blade elements counted with unrestricted global sign" (84 × 2, since
fixing the low-index sign to +1 is a labeling convention, not a
mathematical restriction) and "unordered zero-divisor pairs" (336 / 2, real
because of bilaterality — see §2). These are different objects that happen
to coincide numerically. **Any external citation of "84" or "168" against
this repo's "336" must be disambiguated before being placed side by side —
this is almost certainly the source of the convention flag raised in
`PRIOR_ART_REVIEW_zda.md` §8.4.8** (now marked resolved, see §5 below).

---

## 2. Bilaterality is Moreno 1997, not a new result

**Correction to an earlier draft of this note:** bilaterality of sedenion
(and higher Cayley-Dickson) zero divisors is a 1997 published result, not
an observation original to this project or to the Phase 5 planning session.

**Citation, read directly from the source** (not taken on trust — the full
PDF was fetched and read):

> G. Moreno, "The zero divisors of the Cayley-Dickson algebras over the
> real numbers," arXiv:q-alg/9710013 (1997; accepted, Bol. Soc. Mat. Mex.).

**Corollary 1.6** (quoted verbatim from the paper): "For x and y in Aₙ,
n ≥ 4. 1. xy = 0 ⇔ yx = 0 ⇔ x̄y = 0 ⇔ xȳ = 0. 2. If x ≠ 0 and xy = 0 then
t(y) = 0. 3. If y ≠ 0 and xy = 0 then t(x) = 0." Immediately followed by:
"we don't have to distinguish between left and right zero divisors because
xy = 0 if and only if yx = 0" — also verbatim. Item 1 is exactly
bilaterality; items 2–3 give the (single-coordinate) purely-imaginary
necessary condition, t(x) = x + x̄ = 2·(real part of x). The stronger
"doubly pure" form — **both** Cayley-Dickson halves individually trace-zero
(matching Reggiani's Prop. 2.1 characterization used elsewhere in this
repo) — is **Corollary 1.9**, not 1.6, proved via the same paper's "ˇx"
half-swap construction (Lemma 1.8).

The theorem holds for **all** Cayley-Dickson algebras Aₙ, n ≥ 4 (Moreno's
indexing: A₀=ℝ, A₁=ℂ, A₂=ℍ, A₃=𝕆, A₄=sedenions), not sedenions specifically.

**Independent corroboration found while reading the source, not solicited:**
the paper's own worked example on p.2 — "let x = e1+e10 and y = e15−e4 in
ℝ¹⁶ = A4" — is the same pair `sedenion_kernel.py`'s exhaustive search finds
first ("(e1 + e10) * (e4 - e15) = 0"), up to an overall sign on the second
factor. Confirms this repo's Baez-convention `cd_mult` agrees with Moreno's
construction, independent of and prior to the Reggiani cross-check already
on record.

**The five-term norm identity claimed in the original plan draft is not
Moreno's literal Corollary 1.5.** The paper states 4 terms:
‖xy‖ = ‖x̄y‖ = ‖xȳ‖ = ‖yx‖. The 5th term (‖x̄ȳ‖) is a one-line extension
(substitute y ↦ ȳ into the same corollary), not a misquote, but the
literal citation is the 4-term form.

**Proof of bilaterality from these facts** (conjugation is an
anti-automorphism, (xy)* = y*x*, plus zero divisors being purely
imaginary so x* = −x, y* = −y): x⊛y = 0 ⇒ 0 = (x⊛y)* = y*⊛x* =
(−y)⊛(−x) = y⊛x. This is a re-derivation of Moreno's Cor. 1.6, not an
independent theorem — presented here as a numerical corroboration of a
1997 published result, not a new finding.

### 2.1 Numerical verification against this repo's tensors

**(a) Conjugation is an anti-automorphism**, `sedenion_kernel.cd_mult` /
`cd_conj`, 2000 random pairs:

max |(xy)\* − y\*x\*| = **0.00e+00** (exact to float64).

**(b) Two-blade zero divisors are purely imaginary on both CD-halves**
(coord 0 and coord 8), all 336 pairs: max|coord0| = max|coord8| =
**0.00e+00** — trivial by construction for two-blades (basis indices are
all ≥ 1), included as a sanity check on the general claim, which rests on
Moreno Cor. 1.9 as a theorem rather than this sample.

**(c) X (per-i shuffled construction) has a left identity only, not
two-sided** — checked directly, `phase4_layers.shuffled_structure_tensor`,
seed 1337 (representative; construction is seed-independent by design,
`L[0] = eye(16)` fixes e0 as the left-multiplication identity for every
seed):

- max|e0⊛x − x| (left identity) = **0.00e+00**
- max|x⊛e0 − x| (right identity) = **1.00e+00**

Confirms: X's e0 is a left identity, never a right one. Since Moreno's
proof of bilaterality routes through conjugation being an
anti-automorphism — which in turn is a standard consequence of the
Cayley-Dickson recursive conjugate `(x1,x2)‾ = (x̄1,−x2)` applied around a
**two-sided** identity — X's failure to have a two-sided identity is
consistent with (though does not by itself force) its failure to satisfy
Moreno's theorem.

### 2.2 Bilaterality at exact zero-divisor pairs (two-blades)

`sedenion_kernel.structure_tensor(16)` (S) and
`phase4_layers.shuffled_structure_tensor(seed)` (X, the repo's only
implemented shuffle — see §4 on the withdrawn "per-k" variant), pinned
seeds 1337/1338/1339:

| tensor | ordered ZD pairs | reverse also zero | bilateral |
|---|---|---|---|
| **S** | 336 | 336 | **100%** |
| X seed 1337 | 96 | 2 | 2.1% |
| X seed 1338 | 92 | 0 | 0.0% |
| X seed 1339 | 66 | 0 | 0.0% |

(These pinned-seed X counts — 96/92/66 — match the repo's own prior
characterization in `phase4_shuffledT_nulls.py` / `PHASE4_reggiani_reading.md`
exactly. They replace the "96–120 pairs" distributional range in the
plan's original draft, which came from non-pinned chat-side RNG draws.)

### 2.3 Moreno's norm identity, checked everywhere (not just at zero)

The stronger and cheaper certificate: Moreno's Cor. 1.5 holds for **all**
x,y, not only pairs on the zero-divisor variety. Checked the ‖xy‖=‖yx‖
slice on 5000 random unit pairs per tensor, pinned seeds:

| tensor | max\|‖xy‖−‖yx‖\| | median relative asymmetry | max relative asymmetry |
|---|---|---|---|
| **S** | 4.44e-16 | 0.00% | 0.00% |
| X seed 1337 | 0.891 | 15.56% | 135.57% |
| X seed 1338 | 0.801 | 15.54% | 137.32% |
| X seed 1339 | 0.837 | 15.21% | 134.12% |

S is symmetric to float64 machine precision, everywhere, no exceptions. X
is asymmetric almost everywhere, with a median relative gap around 15%
regardless of seed. **This is the preferred X-inequivalence certificate
going forward**: it holds on a full-measure set rather than only the
14-dimensional zero locus, it is a single `einsum` + norm comparison, and
it cites directly to Moreno (1997) rather than needing a fresh defense. It
supersedes the certificate line of work proposed in
`PRIOR_ART_REVIEW_zda.md` §3.1(a) (Reggiani-isometry-type framing) as the
operative check — that framing was about the null-set's smoothness and
remains valid on its own terms (`phase4_reggiani_certificate.py`), but this
identity is cheaper, holds everywhere, and needs no smoothness machinery.

### 2.4 General (non-two-blade) unit vectors

Method: for fixed unit x, the inner minimization over unit y of ‖x⊛y‖ is
exact via SVD (`M_x[m,j] = Σᵢ T[m,i,j]xᵢ`, minimizer is the smallest
right-singular vector, minimum is the smallest singular value — no
iteration needed on y). Outer minimization over x: Adam warm-start (250
steps) refined with L-BFGS (strong-Wolfe line search, 80 steps), 4 restarts
× 10 points per tensor.

| tensor | fwd ‖x⊛y‖ median | rev ‖y⊛x‖ median | fwd max | rev max |
|---|---|---|---|---|
| **S** | 4.54e-11 | 4.539e-11 | 6.78e-11 | 6.776e-11 |
| X seed 1337 | 7.69e-12 | 0.959 | 1.00e-11 | 1.304 |
| X seed 1338 | 7.06e-12 | 0.982 | 1.26e-11 | 1.243 |
| X seed 1339 | 6.99e-12 | 1.028 | 1.97e-11 | 1.267 |

Random-pair baseline (S tensor, uniform random unit pairs, not located on
the variety): mean ‖x⊛y‖ = 0.998, median = 1.001.

For S, forward and reverse residuals agree to 3 significant figures at
every seed-independent restart — bilateral at general points, not just at
basis two-blades, to the precision the optimizer reaches. This did **not**
reach the ~1e-16 a chat-side session reported (this run's optimizer is
less aggressive); given §2's proof, this gap is optimizer convergence, not
a live question about the mathematics — Moreno's Cor. 1.6 is exact, and
this table is a numerical corroboration, not the primary evidence. Not
pursued further, per the plan's own re-prioritization once the theorem was
identified.

For X, forward residuals converge to ~1e-11 (points genuinely on or very
near the one-sided zero-divisor variety) while reverse residuals are
statistically indistinguishable from the random-pair baseline (0.96–1.03
vs. 0.998–1.001). X's zero-divisor variety is a real, locatable object;
its bilaterality is not — reversing lands you nowhere near it.

---

## 3. Consequence: S_sym is not a new variant

If ‖P⊛Q‖ = ‖Q⊛P‖ identically for S (§2.3, confirmed to 4.4e-16 over 5000
pairs, not merely on the zero set), then the symmetrized score proposed in
`PHASE5_PLAN.md` §2.2,

  ‖P⊛Q‖ + ‖Q⊛P‖ = 2‖P⊛Q‖,

is an exact constant multiple of S's existing score, not a new function of
(P,Q). Since γ is a learned per-head scalar multiplying the score (spec
§2, init 1.0), training "S_sym" measures γ-initialization sensitivity
(effectively γ init 2.0), not algebra. **S_sym is withdrawn as a variant**;
see `PHASE5_PLAN.md` §4/§2.2/§5/§8 for the corresponding plan edits. X_sym
is unaffected and remains in the design — X's two conditions
(‖P⊛Q‖≈0, ‖Q⊛P‖≈0) are independent (§2.3's ~15% median asymmetry, ~1.0
median reverse-norm), so symmetrizing X is a real, non-redundant test.

---

## 4. Correction: the "per-k shuffle" row is withdrawn

The plan's original §2 table included a row for "X (per-k shuffle, 3
draws): 618–814 ordered pairs" alongside the "per-i shuffle" row. Only one
shuffled-tensor construction exists in this codebase —
`phase4_layers.shuffled_structure_tensor` / its numpy mirror in
`phase4_shuffledT_nulls.py` — and it is the per-i construction (each
basis index i ≥ 1 draws one independent random signed-permutation matrix
for left multiplication). The "per-k" row does not correspond to any code
in this repo and cannot be verified against it. Per the session that
authored the original plan: this was a chat-side artifact of an ambiguous
handoff description ("15 independent permutations + independent signs,
one per k"), where both readings were tested for robustness and both ended
up written into the table as if both were implemented. **Withdrawn, not
retained as a labeled paper-only row** — it was never implemented anywhere,
so there is nothing to label.

---

## 5. Prior-art convention flag — resolved

`PRIOR_ART_REVIEW_zda.md` §8.4.8 flagged sources reporting "84 rather than
336" sedenion zero-divisor pairs and exemplar pairs in an unfamiliar form,
without reconciling the discrepancy. §1 above resolves it: 84 and 336 are
both correct, simultaneously, as counts of different objects in the same
lattice (84 = signed two-blade elements acting as a left factor; 336 =
ordered pairs). The 168 collision in §1 is almost certainly why sources
disagree even when internally consistent — "168" denotes two different
things depending on whether you're counting elements-with-free-sign or
unordered pairs. §8.4.8 updated to RESOLVED, citing this note.

---

## 6. What remains open

- The general-vector localization (§2.4) was not pushed to machine
  precision; not pursued further since §2 makes it corroboration of a
  theorem rather than primary evidence, per the plan's own
  re-prioritization.
- §1's "independent from-scratch Cayley-Dickson construction" (a parallel
  session's cross-check using a different implementation) was not
  re-implemented and independently re-run here; the counting lattice was
  instead verified directly against `sedenion_kernel.py`, which is
  sufficient to confirm the lattice against this repo but does not by
  itself confirm the parallel session's own independent implementation was
  bug-free. Not treated as a gap in the *lattice's* correctness (three
  converging routes, per §1), only flagged for precision about what was
  and wasn't rerun.
- Moreno's paper is cited throughout the later Cayley-Dickson zero-divisor
  literature already on record in this repo (Reggiani cites it; see
  `PHASE4_reggiani_reading.md`), so this is very unlikely to be an isolated
  citation, but no independent forward-citation check of Moreno 1997 itself
  was performed as part of this note.
