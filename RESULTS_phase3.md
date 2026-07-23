# RESULTS — Phase 3: 18-run TinyStories grid (2026-07-19)

Full pre-registered grid per `ZDA_experiment_spec.md`: 6 variants × 3 seeds
(1337/1338/1339), TinyStories, 300M tokens per run (18,311 steps × 16,384
tokens), train ctx 256, d_model 384 (B1: 432), 6 layers / 6 heads, ~12.2M
params (B1: 15.2M, FLOP-matched to V1), vocab 4096 BPE. Run via `zda_grid.py`
on Colab (notebook `ZDA_Phase3_TinyStories_Grid`); outputs in `zda_runs_ts/`
(local copy of Drive `zda_runs_ts/`, checkpoints left on Drive). Preflight
checks P1–P3 passed on CUDA before launch, including step-0 V2 loss == B0
loss with diff exactly 0.0. Total compute ≈ 7.9 GPU-hours (one run on T4,
seventeen on A100; see §6).

**Verdict: all pre-registered hypotheses are negative.** The ZD gate went
unused, nothing observed is specific to zero-divisor or sedenion structure,
and PHM-16 loses in-distribution at matched parameters. One exploratory
finding survives with honest attribution: 16-way weight-sharing (real or
shuffled) trades a small in-distribution loss increase for a ~3× length-
generalization improvement — an effect of parameter sharing generally, not
of the algebra.

## 1. Grand summary (mean ± std over 3 seeds)

```
variant   n    val loss (mean ± std)   ppl@512  ppl@1024
B0        3         1.5631 ± 0.0057      9.24     25.40
B1        3         1.5319 ± 0.0071      8.63     24.47
V1        3         1.6283 ± 0.0081      5.24      8.21
V2        3         1.5630 ± 0.0058      9.21     25.39
V3        3         1.5632 ± 0.0057      9.23     25.28
V4        3         1.6307 ± 0.0042      5.24      8.80
```

Per-run finals (val loss; ppl at eval ctx 256/512/1024; wall seconds):

| run | val | ppl@256 | ppl@512 | ppl@1024 | wall_s |
|---|---|---|---|---|---|
| B0_seed1337 | 1.5613 | 4.821 | 9.278 | 24.329 | 10164 |
| B0_seed1338 | 1.5695 | 4.863 | 9.003 | 26.333 | 888 |
| B0_seed1339 | 1.5586 | 4.848 | 9.428 | 25.537 | 888 |
| B1_seed1337 | 1.5317 | 4.676 | 9.078 | 26.688 | 1018 |
| B1_seed1338 | 1.5390 | 4.708 | 8.792 | 23.817 | 1018 |
| B1_seed1339 | 1.5249 | 4.675 | 8.006 | 22.891 | 1018 |
| V1_seed1337 | 1.6319 | 5.143 | 5.228 | 7.728 | 1111 |
| V1_seed1338 | 1.6340 | 5.175 | 5.159 | 8.280 | 1112 |
| V1_seed1339 | 1.6191 | 5.141 | 5.323 | 8.622 | 1112 |
| V2_seed1337 | 1.5613 | 4.820 | 9.125 | 23.843 | 1124 |
| V2_seed1338 | 1.5695 | 4.856 | 9.029 | 26.429 | 1124 |
| V2_seed1339 | 1.5583 | 4.842 | 9.486 | 25.908 | 1125 |
| V3_seed1337 | 1.5614 | 4.820 | 9.140 | 23.861 | 1125 |
| V3_seed1338 | 1.5696 | 4.865 | 9.080 | 26.200 | 1124 |
| V3_seed1339 | 1.5587 | 4.849 | 9.463 | 25.766 | 1125 |
| V4_seed1337 | 1.6303 | 5.142 | 5.301 | 9.246 | 1113 |
| V4_seed1338 | 1.6351 | 5.177 | 5.099 | 8.181 | 1113 |
| V4_seed1339 | 1.6268 | 5.180 | 5.329 | 8.967 | 1114 |

Within each seed, B0/V2/V3 val losses agree to the third decimal (e.g. seed
1337: 1.5613 / 1.5613 / 1.5614) — the gated variants are functionally the
baseline.

## 2. Pre-registered decision rules (spec §1) — outcomes

Grid output, verbatim:

```
H2  (V2 must beat B1): V2 - B1 = +0.0311, 2x pooled std = 0.0129 -> negative
H3g (V2 vs gate control; ~0 => not ZD-specific): V2 - V3 = -0.0002, 2x pooled std = 0.0115 -> negative
H1  (V1 vs param-matched baseline): V1 - B0 = +0.0652, 2x pooled std = 0.0140 -> negative
H3s (V1 vs structure control; ~0 => not algebra): V1 - V4 = -0.0024, 2x pooled std = 0.0128 -> negative
```

- **H1 — negative.** V1 (PHM-16) is 0.065 nats *worse* than the
  param-matched dense baseline in-distribution; the spec required equal or
  better.
- **H2 — negative on both arms.** Spec H2 allows a win on validation loss at
  matched FLOPs *or* on length generalization. Val loss: V2 loses to B1 by
  +0.0311 (threshold 0.0129). Length generalization: V2 ppl@1024 = 25.39 vs
  B0 25.40 and B1 24.47 — no improvement whatsoever (V2 is
  indistinguishable from ungated attention at every eval length).
- **H3 — negative (specificity absent).** V2 − V3 = −0.0002: the ZD frame
  and a random orthonormal frame produce statistically identical models.
  V1 − V4 = −0.0024: the true sedenion multiplication tensor and a shuffled
  one produce statistically identical models. Nothing in this experiment
  distinguishes zero-divisor/sedenion structure from its matched controls.

Falsification condition from spec §1 ("if V2 ≈ V3, any gain is not
ZD-specific") is met — moot in practice, since there was no gain to
attribute.

## 3. Gate diagnostics — β trajectories (spec §5)

From `eval_log.jsonl` (per-head β at every eval; 36 heads/run, 108
heads/variant pooled):

| run | final max\|β\| | final mean β | pos/neg | traj max\|β\| | @step | final Σ grad-norm |
|---|---|---|---|---|---|---|
| V2_seed1337 | 0.0536 | +0.0034 | 30/6 | 0.0536 | 18311 | 7.5e-03 |
| V2_seed1338 | 0.0923 | +0.0048 | 28/8 | 0.0972 | 10682 | 8.0e-03 |
| V2_seed1339 | 0.0244 | +0.0020 | 25/11 | 0.0344 | 3052 | 7.6e-03 |
| V3_seed1337 | 0.2342 | +0.0007 | 22/14 | 0.2479 | 10682 | 6.9e-03 |
| V3_seed1338 | 0.3516 | +0.0114 | 24/12 | 0.3636 | 9156 | 8.9e-03 |
| V3_seed1339 | 0.2339 | −0.0052 | 26/10 | 0.2482 | 10682 | 8.0e-03 |

Pooled final β: V2 median |β| 0.0020, p90 0.0093, max 0.0923 (83 pos / 25
neg); V3 median |β| 0.0037, p90 0.0606, max 0.3516 (72 pos / 36 neg).

Four observations:

1. **The gate was weakly recruited but functionally inert.** β gradient
   norms were nonzero at every eval through end of training, so this is SGD
   actively pricing the direction, not a dead parameter. And it was not
   wholly declined: in every gated run a few heads show monotone,
   saturating β growth (e.g. V2_seed1338 L0h1 climbs steadily to +0.092 by
   step ~9k and holds; V2_seed1337 L0h2 to +0.054). But even the largest V2
   gate shifts logits by ≲ 0.1·|a+c| — perturbative at softmax scale —
   consistent with V2 matching B0 to the third decimal. β is unconstrained
   in sign by design (negative = amplify annihilation-aligned pairs); the
   model recruited neither suppression nor amplification to any
   loss-relevant degree.
2. **Recruitment concentrates almost entirely in layer 0, in all six gated
   runs.** Final mean|β| per layer (×1000):

   | run | L0 | L1 | L2 | L3 | L4 | L5 |
   |---|---|---|---|---|---|---|
   | V2_seed1337 | 16.2 | 1.2 | 1.2 | 1.9 | 2.6 | 2.5 |
   | V2_seed1338 | 26.8 | 0.9 | 2.8 | 1.5 | 2.2 | 2.8 |
   | V2_seed1339 | 8.7 | 1.0 | 2.2 | 3.5 | 3.2 | 1.7 |
   | V3_seed1337 | 87.8 | 4.3 | 3.1 | 5.3 | 4.3 | 4.3 |
   | V3_seed1338 | 120.0 | 2.9 | 3.4 | 3.5 | 6.8 | 10.6 |
   | V3_seed1339 | 97.0 | 3.4 | 2.9 | 3.1 | 8.1 | 2.4 |

   Layer 0 exceeds every deeper layer by roughly 5–30×; layers 1–5 sit at
   noise scale (mean|β| ≈ 0.001–0.003) with frequent sign flips across
   evals (wandering around zero). Descriptive only — but seed-stable, and
   it localizes what little use the gate found to attention over raw token
   embeddings. Layer-differentiated per-head attention dials are not
   unique to this setup: Correia, Niculae & Martins (adaptively sparse
   transformers, α-entmax, EMNLP 2019) report a learned per-head sparsity
   parameter that differentiates by layer with no accuracy cost — the same
   qualitative pattern, for a dial that (unlike β) *was* recruited to
   material effect. That precedent both deflates this layer-0
   concentration's novelty (it looks like a general property of per-head
   attention dials, not something specific to ZD structure) and supports
   its reality (it replicates a known pattern rather than being an
   artifact of this setup). It also sharpens observation 1's contrast:
   α-entmax's dial modifies normalization directly (cheap, global, always
   relevant); β adds a bias contingent on a specific feature (|a_i+c_j|)
   being useful — the "load-bearing" distinction this experiment's null
   turns on, in another guise (`PRIOR_ART_REVIEW_zda.md` §4).
3. **The random frame nominally recruited harder than the ZD frame — with
   a scale caveat.** V3's largest heads reach |β| ≈ 0.23–0.36 versus V2's
   ≤ 0.10, and V3's strongest heads are mixed-sign (e.g. seed 1337:
   L0h3 = −0.234 alongside L0h1 = +0.089) while V2's recruited heads are
   uniformly on the suppression side. Raw β is not directly comparable
   across frames, however: the gate applies β·|a_i + c_j|, and different
   frames induce different |a_i + c_j| activation scales, so β partially
   absorbs a scale factor. The conservative reading is directional only:
   the ZD frame showed no *greater* engagement than a random one — H3g's
   conclusion restated from the parameter side rather than the loss side.
4. **Sign split of pooled final β is mildly positive (suppression side) in
   both variants**, ~3:1 for V2 (83/25) and 2:1 for V3 (72/36). Outside
   the layer-0 heads noted above, magnitudes are at noise scale. No
   amplification story is hiding in the V2 traces; V3's largest negative
   heads (−0.23 twice, −0.13) show SGD will use the amplification side of
   an arbitrary frame, to no benefit.

Remaining spec §5 diagnostics require the checkpoints, which remain on
Drive. The |a_i + c_j| activation distributions are the one computation
that would firm up observation 3 (cross-frame effective gate strength);
attention entropy per head is optional given the null primary results and
perturbative gate magnitudes.

## 4. Exploratory findings (not pre-registered, labeled as such)

- **Weight-sharing buys length generalization.** Both PHM variants (V1 real
  tensor, V4 shuffled) hold ppl ≈ 5.2 at ctx 512 (vs ≈ 4.8–5.2 at the
  training ctx 256 — essentially no degradation) and ppl ≈ 8.2–8.8 at ctx
  1024, where every dense variant degrades to ≈ 23–27. That is a ~3× OOD
  perplexity advantage at 4× training length, paid for with +0.065–0.068
  nats in-distribution. Because V4 (shuffled tensor, no algebraic meaning)
  matches V1, the effect is attributable to 16-way parameter sharing /
  reduced projection rank generally — **not** to sedenion structure.
  **Variance caveat:** Zhou et al. ("Transformers Can Achieve Length
  Generalization But Not Robustly," arXiv:2402.09371) report that length
  generalization is fragile and significantly influenced by random
  initialization and data order, with large cross-run variance. This
  result is 3 seeds; the *direction* (V1≈V4 ahead of dense) is consistent
  across all three and the V1-vs-V4 attribution is safe (both arms share
  any such confound identically), but the **magnitude** (~3×) should be
  read with that fragility in mind rather than as a tight estimate
  (`PRIOR_ART_REVIEW_zda.md` §5.1).
- **No reliable V1-over-V4 residual.** V1 beats V4 at ppl@1024 at seeds 1337
  (7.73 vs 9.25) and 1339 (8.62 vs 8.97) but loses at 1338 (8.28 vs 8.18);
  the pooled gap (8.21 vs 8.80) is within seed-to-seed spread. No
  algebra-specific length-generalization claim is supported.
- **B1's extra parameters help in-distribution (−0.031 nats vs B0) but not
  out-of-length** (ppl@1024 24.5 vs 25.4 — marginal).

## 5. Conclusion

At this scale and task, bilateral zero-divisor annihilation structure adds
no measurable value inside a transformer, either as an attention gate (V2)
or via sedenion-structured projections (V1). The gate result is
particularly clean: with a live gradient and an unconstrained sign, across
108 heads × 3 seeds, SGD sampled the ZD gate — a few layer-0 heads grew β
monotonically to saturation — and priced it at a perturbative magnitude
that never touched the loss, while a random-frame control engaged at least
as strongly (nominally more, modulo feature scale) for identically null
benefit. The gate was not ignored for lack of gradient; it was tried and
found nearly worthless at the margin against an already-sufficient
attention pathway. The
one effect worth carrying forward — the weight-sharing length-
generalization trade — belongs to the PHM family generally and is already
known territory (Zhang et al. PHM); our contribution is the shuffled-tensor
control showing the algebra contributes nothing on top.

Per spec §8, this is the honest framing: the experiment was designed to
falsify cheaply, and it did. Any future ZD-structure claim needs a
mechanism by which the annihilation manifold could matter to the task —
this experiment found none at 12M params / 300M tokens.

## 6. Deviations & provenance notes

- **ZD frame is Pattern 2 (e3+e12, e5+e10)** — changed from the original
  (e1+e10, e4−e15) before launch; rationale, verification, and Lean 4 proof
  reference logged in `RESULTS_phase2_smoke.md` §6. All 18 runs used
  Pattern 2.
- **Hardware split:** B0_seed1337 trained on a T4 (FP32 matmuls, ~10,164s);
  the remaining 17 runs on an A100 (TF32 matmuls, ~888–1,125s). Data order
  is identical across variants at a seed regardless of device; the B0/V2/V3
  agreement to the third decimal at seed 1337 (T4 B0 vs A100 V2/V3) shows
  the numeric difference is far below seed noise.
- **B1_seed1337 trained twice:** a T4 partial run (to step ~3052) was lost
  when the VM died before `ckpt_last.pt` synced to Drive; the recorded run
  is a clean from-scratch A100 run. No partial-resume artifacts are in the
  final data. (The mid-run resume path itself is tested by
  `--sim_preempt_step` and was not needed in the final grid.)
- A frontend disconnect during B1_seed1339 did not interrupt the backend;
  the run continued and completed normally.
