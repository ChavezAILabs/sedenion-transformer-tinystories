# Experiment Spec — Zero-Divisor Attention (ZDA) at Toy Scale

**Chavez AI Labs | Applied Pathological Mathematics**
**Goal:** Determine, at nanoGPT scale and consumer-GPU cost, whether sedenion algebra structure — specifically bilateral zero divisor annihilation — adds measurable value inside a transformer, with attention as the primary target.

---

## 1. Hypotheses (pre-registered)

- **H1 — Structured projections.** Constraining attention projection matrices (Q, K, V, O) to sedenion-multiplication structure (a PHM-16 layer) achieves equal or better validation loss than an unconstrained baseline at **matched parameter count**.
- **H2 — ZD-gated attention (the novel claim).** Adding a zero-divisor annihilation gate to attention logits improves validation loss at **matched FLOPs**, or improves **length generalization**, versus standard attention.
- **H3 — Specificity.** Any gains from H1/H2 depend on the *sedenion/zero-divisor* structure specifically, not on generic parameter sharing or generic gating (tested via controls V3/V4 below).

**Falsification, stated up front:** if V2 does not beat B1 by more than 2× the pooled 3-seed standard deviation, H2 is negative. If V2 ≈ V3, any gain is not ZD-specific. A negative result gets written up too — that is the point of doing this at $100 instead of at scale.

---

## 2. The mechanism, precisely

### 2.1 Verified algebra kernel (done — see `sedenion_kernel.py`)

The structure tensor `T ∈ R^{16×16×16}` with `(x·y)_k = Σ_ij T[k,i,j] x_i y_j` is built from the Cayley-Dickson construction (Baez convention) and verified:

- Reproduces recursive multiplication to ~1e-16.
- Norm multiplicative at 4D/8D (division algebras), non-multiplicative at 16D — zero divisors exist.
- Exhaustive search over `(e_i ± e_j)(e_k ± e_l)` finds **336 annihilating ordered products**.
- **Bilateral collapse identity `(aP + bQ)(bP + cQ) = −2b(a+c)e₀` held numerically for 336/336 found pairs** — annihilation occurs exactly when `a = −c`. This is the gate primitive.

T is a fixed sparse ±1 tensor (256 nonzeros). All downstream math is real-valued einsums — no exotic numerics.

### 2.2 V1 — PHM-16 projections

Replace each `nn.Linear(d, d)` in attention with a parameterized hypercomplex multiplication layer at n = 16 (Zhang et al. 2021 style):

```
W = Σ_{k=0..15}  L_k ⊗ A_k
```

where `L_k ∈ R^{16×16}` is the fixed left-multiplication matrix of basis element `e_k` (slices of T), and `A_k ∈ R^{(d/16)×(d/16)}` are learned. Parameter count per projection drops ~16× ; recover budget by widening d (see matching table).

### 2.3 V2 — ZD-gate attention (primary novelty)

Per head, in addition to standard dot-product logits:

1. Learn (or freeze, ablation) a zero-divisor frame: vectors `p, q ∈ R^{d_h}` initialized from a verified annihilating pair embedded in the head dimension (d_h = 64 = 4 sedenion blocks; frame lives in one designated block).
2. Compute per-token coefficients: `a_i = ⟨q_vec_i, p⟩`, `c_j = ⟨k_vec_j, q⟩`, and a learned per-head scalar `b`.
3. By the collapse identity, the product magnitude of the induced bilateral pair is `2|b|·|a_i + c_j|`. Gate the logits:

```
logits_ij = (q_i · k_j)/√d_h  −  β · |a_i + c_j|        (β ≥ 0 learned, init 0)
```

**Interpretation:** query–key pairs satisfying the annihilation resonance `a_i ≈ −c_j` pass through a null channel unpenalized; incompatible pairs are suppressed proportionally to how far they sit from the zero-divisor manifold. This is an O(n²) additive term — same asymptotic cost as attention itself, negligible FLOP overhead (one rank-1 outer sum per head).

β initialized to 0 makes V2 exactly equal to baseline at step 0 — the model must *choose* to use the gate, and learned β trajectories are themselves a diagnostic.

### 2.4 Controls

- **V3 (gate control):** identical gate functional form, `p, q` drawn random-orthogonal (not a zero-divisor pair), frozen. If V2 ≈ V3, the win is "additive gating helps," not "zero divisors help."
- **V4 (structure control):** PHM-16 with a shuffled structure tensor (random signed permutation tensor, same sparsity as T). If V1 ≈ V4, the win is "Kronecker parameter sharing helps," not "sedenion algebra helps." (Known risk: PHM literature suggests generic structure captures most of the benefit — this control is where H1 most likely dies, and that's fine.)

---

## 3. Variant grid

| ID | Description | What it tests |
|----|-------------|---------------|
| B0 | nanoGPT baseline, param-matched | reference |
| B1 | nanoGPT baseline, FLOP-matched (wider) | fair fight for V1/V2 |
| V1 | PHM-16 Q/K/V/O projections | H1 |
| V2 | Standard projections + ZD-gate | H2 (primary) |
| V3 | V2 with random frozen frame | H3 gate specificity |
| V4 | V1 with shuffled structure tensor | H3 structure specificity |

6 variants × 3 seeds = **18 runs**.

---

## 4. Model, data, training

**Architecture (all variants):** decoder-only transformer, 6 layers, 6 heads, d_model = 384, d_h = 64, **RoPE positional encoding** (required — learned positions can't do the length-generalization eval), context 256, ~10.6M params nominal. Widths adjusted per variant to hit matching targets: params within ±1%, FLOPs within ±5%; publish the exact table with the results.

**Data:**
- Phase A smoke test: character-level Tiny Shakespeare (~1M chars). Purpose: catch bugs, not draw conclusions.
- Phase B main run: **TinyStories** (HuggingFace `roneneldan/TinyStories`), 4k-vocab BPE trained on the corpus. TinyStories gives coherent-language signal at 10M params, which OpenWebText does not.

**Training:** AdamW (β=0.9/0.95, wd 0.1), lr 3e-4, cosine decay, 2k warmup steps, grad clip 1.0, fixed budget **300M tokens** per run (~1–2 h per run on a single RTX 3090/4090 or Colab A100). Identical data order per seed across variants (seed the dataloader, not just the model).

---

## 5. Metrics

**Primary:** validation loss at the fixed 300M-token budget, mean ± std over 3 seeds; and the full loss-vs-FLOPs curve (checkpoint evals every 25M tokens).

**Secondary (H2):** length generalization — perplexity at eval context 512 and 1024 (train ctx 256, RoPE makes this meaningful). Hypothesis-relevant because a structural compatibility score could plausibly transfer across lengths better than learned attention patterns.

**Diagnostics:**
- β trajectory per head/layer (β → 0 everywhere = the gate is dead weight; report honestly).
- Distribution of `|a_i + c_j|` over trained model activations — does the model organize tokens onto the annihilation manifold?
- Attention entropy per head, V2 vs B1.
- Gradient norm through the gate (check it trains at all).

---

## 6. Implementation plan

| Phase | Deliverable | Effort |
|-------|-------------|--------|
| 0 | `sedenion_kernel.py` — structure tensor + verification | **DONE** (all 6 checks pass) |
| 1 | `zda_layers.py` — PHM-16 linear + ZD-gate head in PyTorch; unit tests: gradcheck, param counts, V2(β=0) ≡ B0 output equality to 1e-6 | 1 day |
| 2 | nanoGPT fork wiring + Shakespeare smoke run (all 6 variants, 1 seed, 10 min each) | 1 day |
| 3 | TinyStories tokenizer + full 18-run grid | 2–4 days wall clock |
| 4 | Analysis notebook + writeup (positive or negative) | 1–2 days |

**Porting note:** the kernel is pure einsum; the PyTorch layer is
`torch.einsum('kij,...i,...j->...k', T, x, y)` with T as a registered buffer. No custom kernels needed at this scale.

## 7. Budget

Single consumer GPU. ~25–40 GPU-hours total. $0 (own hardware) to ~$50–100 (cloud). No excuse not to run it.

## 8. Known risks and honest framings

1. **PHM-16 may underperform at tiny scale** — the 16× parameter constraint bites hardest when models are small. A V1 loss at 10M params doesn't kill the idea at 100M; note it, don't over-read it.
2. **The gate may be learned away (β→0).** That is a valid negative result for H2 as formulated; the follow-up would be moving the gate into value mixing rather than logits before abandoning it.
3. **V3/V4 controls are the real judges.** The most likely positive-ish outcome is "structure helps, sedenion-specific structure doesn't." Decide now that you'll report that as what it is.
4. **One result to cite either way:** this slots into the hypercomplex-networks literature (quaternion transformers, PHM layers, Clifford networks). A clean 18-run ablation with controls is publishable as a workshop paper regardless of sign — and it's the artifact that makes the larger integration conversation credible.

## 9. What success unlocks (and what it doesn't)

A V2 win over B1 and V3 at matched FLOPs, replicated across seeds, is evidence that zero-divisor structure does real work in attention — the first load-bearing brick for the larger program. It does **not** yet justify acquiring large compute; the next step would be the same ablation at 50–125M params on OpenWebText, still under $1k. Scale purchases come after the mechanism survives two rounds of fair fights.
