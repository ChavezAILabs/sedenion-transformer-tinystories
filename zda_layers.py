"""
zda_layers.py — Phase 1 of the ZDA experiment (see ZDA_experiment_spec.md)

PyTorch layers implementing:
  * PHMLinear      — linear layer constrained to sedenion (16D) multiplication
                     structure. W = sum_k L_k (x) S_k with L_k the FIXED signed
                     permutation matrices of left-multiplication by sedenion
                     basis elements, S_k learned.  Params = in*out/16.
                     Mathematically: a matrix of LEARNED SEDENIONS acting by
                     left multiplication (verified in test_zda_numpy.py).
  * ZDGatedCausalSelfAttention — standard causal MHA + RoPE, plus the
                     zero-divisor annihilation gate:
                        logits_ij = (q_i . k_j)/sqrt(d_h)  -  beta * |a_i + c_j|
                     where a_i = <q_i, p_hat>, c_j = <k_j, q_hat> and (P, Q)
                     is a VERIFIED annihilating sedenion pair (default
                     P = e3 + e12, Q = e5 + e10; bilateral P*Q = Q*P = 0
                     under the Baez convention, cross-checked against the
                     owner's CAILculator/KSJ line where this pair —
                     "Pattern 2" — is the unique Canonical Six pair that
                     is also Clifford-bilateral).
                     beta is initialized to 0, so at init the layer is
                     EXACTLY standard attention (tested below); the model
                     must recruit the gate via gradient descent.

Variant mapping (spec section 3):
  B0/B1 : ZDGatedCausalSelfAttention(gate=False, proj="dense")
  V1    : gate=False, proj="phm16"
  V2    : gate=True,  frame="zd",     proj="dense"     <- primary
  V3    : gate=True,  frame="random", proj="dense"     <- gate control
  V4    : gate=False, proj="phm16_shuffled"            <- structure control

Design deviations from spec (documented):
  * beta is UNCONSTRAINED (not softplus/>=0). Reason: any smooth
    reparameterization pinned to 0 at init (e.g. beta^2) has zero gradient
    at init and the gate could never turn on; softplus cannot represent
    exact 0. Plain beta gives exact baseline equality at init AND live
    gradient. Sign of learned beta is itself a diagnostic (positive =
    ZD-resonance attraction, negative = repulsion).

Run the embedded tests on a machine with torch installed:
    python zda_layers.py
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ALG = 16  # sedenion dimension

# ----------------------------------------------------------------------
# Cayley-Dickson structure (Baez convention), mirrored from
# sedenion_kernel.py which was verified against exhaustive ZD search.
# ----------------------------------------------------------------------

def _cd_conj(x: torch.Tensor) -> torch.Tensor:
    out = -x.clone()
    out[0] = x[0]
    return out

def _cd_mult(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    if n == 1:
        return x * y
    h = n // 2
    a, b = x[:h], x[h:]
    c, d = y[:h], y[h:]
    real = _cd_mult(a, c) - _cd_mult(_cd_conj(d), b)
    imag = _cd_mult(d, a) + _cd_mult(b, _cd_conj(c))
    return torch.cat([real, imag])

def structure_tensor(dim: int = N_ALG, dtype=torch.float64) -> torch.Tensor:
    """T[k, i, j] with (x*y)_k = sum_ij T[k,i,j] x_i y_j."""
    T = torch.zeros(dim, dim, dim, dtype=dtype)
    I = torch.eye(dim, dtype=dtype)
    for i in range(dim):
        for j in range(dim):
            T[:, i, j] = _cd_mult(I[i], I[j])
    return T

def left_mult_matrices(T: torch.Tensor) -> torch.Tensor:
    """L[k][m, j] = T[m, k, j]; (e_k * x)_m = sum_j L[k][m,j] x_j.
    Each L_k is a signed permutation matrix; L_0 = I."""
    return T.permute(1, 0, 2).contiguous()

def shuffled_left_mult_matrices(generator: torch.Generator,
                                dim: int = N_ALG) -> torch.Tensor:
    """V4 control: random signed permutation matrices with L_0 = I
    (preserving the unit-element property, randomizing the algebra)."""
    L = torch.zeros(dim, dim, dim, dtype=torch.float64)
    L[0] = torch.eye(dim, dtype=torch.float64)
    for k in range(1, dim):
        perm = torch.randperm(dim, generator=generator)
        signs = torch.randint(0, 2, (dim,), generator=generator) * 2 - 1
        L[k, torch.arange(dim), perm] = signs.double()
    return L

# Verified annihilating pair: (e3 + e12) * (e5 + e10) = 0 in BOTH orders
# under this repo's Baez convention (verified 2026-07-17, worst-case
# collapse-identity error 4e-16). Chosen over the original (e1+e10, e4-e15)
# because it is KSJ "Pattern 2" — the unique Canonical Six pair that is
# bilateral in both the Cayley-Dickson and Clifford frameworks (AIEX-725).
# The collapse identity (aP + bQ)(bP + cQ) = -2 b (a+c) e0 holds for it.
ZD_PAIR = ((3, +1.0, 12), (5, +1.0, 10))


# ----------------------------------------------------------------------
# PHM-16 linear layer
# ----------------------------------------------------------------------

class PHMLinear(nn.Module):
    """W = sum_k L_k (kron) S_k.  Feature layout is COMPONENT-MAJOR:
    flat index (a * blocks + c) = sedenion component a of block c.
    in_features and out_features must be divisible by 16."""

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, L: torch.Tensor | None = None):
        super().__init__()
        assert in_features % N_ALG == 0 and out_features % N_ALG == 0
        self.in_b = in_features // N_ALG
        self.out_b = out_features // N_ALG
        if L is None:
            L = left_mult_matrices(structure_tensor())
        self.register_buffer("L", L.float())                     # (16,16,16)
        self.S = nn.Parameter(torch.empty(N_ALG, self.out_b, self.in_b))
        # Kaiming-style init scaled for the kron sum:
        # each output unit sums over 16 * in_b inputs with |L| entries = 1.
        nn.init.normal_(self.S, std=1.0 / math.sqrt(N_ALG * self.in_b))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def weight(self) -> torch.Tensor:
        # W[(a,c),(b,d)] = sum_k L[k,a,b] S[k,c,d]
        W = torch.einsum("kab,kcd->acbd", self.L, self.S)
        return W.reshape(N_ALG * self.out_b, N_ALG * self.in_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight(), self.bias)

    def extra_repr(self) -> str:
        return (f"in={N_ALG * self.in_b}, out={N_ALG * self.out_b}, "
                f"params={self.S.numel()} (dense equiv /16)")


def make_proj(kind: str, d_in: int, d_out: int, bias: bool,
              gen: torch.Generator | None = None) -> nn.Module:
    if kind == "dense":
        return nn.Linear(d_in, d_out, bias=bias)
    if kind == "phm16":
        return PHMLinear(d_in, d_out, bias=bias)
    if kind == "phm16_shuffled":
        assert gen is not None, "pass a torch.Generator for reproducibility"
        return PHMLinear(d_in, d_out, bias=bias,
                         L=shuffled_left_mult_matrices(gen))
    raise ValueError(kind)


# ----------------------------------------------------------------------
# RoPE
# ----------------------------------------------------------------------

def rope_cache(seq_len: int, d_head: int, device, base: float = 10000.0):
    assert d_head % 2 == 0
    half = d_head // 2
    freqs = base ** (-torch.arange(0, half, device=device).float() / half)
    t = torch.arange(seq_len, device=device).float()
    ang = torch.outer(t, freqs)                       # (T, half)
    return ang.cos(), ang.sin()

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # x: (B, H, T, D). Pairing convention: (x_even, x_odd) rotated jointly.
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin,
                        x1 * sin + x2 * cos], dim=-1).flatten(-2)


# ----------------------------------------------------------------------
# ZD-gated causal self-attention
# ----------------------------------------------------------------------

class ZDGatedCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 gate: bool = True, frame: str = "zd",
                 learnable_frame: bool = False,
                 proj: str = "dense", dropout: float = 0.0,
                 seed: int = 0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d_h = d_model // n_heads
        assert self.d_h >= N_ALG, "head dim must hold one sedenion block"
        self.gate = gate
        self.dropout = dropout

        gen = torch.Generator().manual_seed(seed)
        self.q_proj = make_proj(proj, d_model, d_model, False, gen)
        self.k_proj = make_proj(proj, d_model, d_model, False, gen)
        self.v_proj = make_proj(proj, d_model, d_model, False, gen)
        self.o_proj = make_proj(proj, d_model, d_model, False, gen)

        if gate:
            p_vec = torch.zeros(self.d_h)
            q_vec = torch.zeros(self.d_h)
            if frame == "zd":
                (i, s1, j), (k, s2, l) = ZD_PAIR
                p_vec[i], p_vec[j] = 1.0, s1      # P = e3 + e12
                q_vec[k], q_vec[l] = 1.0, s2      # Q = e5 + e10
            elif frame == "random":
                g = torch.Generator().manual_seed(seed + 1)
                p_vec = torch.randn(self.d_h, generator=g)
                q_vec = torch.randn(self.d_h, generator=g)
            else:
                raise ValueError(frame)
            p_vec = p_vec / p_vec.norm()
            q_vec = q_vec / q_vec.norm()
            # one frame per head (broadcast copies; independent if learnable)
            P = p_vec.repeat(n_heads, 1)
            Q = q_vec.repeat(n_heads, 1)
            if learnable_frame:
                self.frame_p = nn.Parameter(P)
                self.frame_q = nn.Parameter(Q)
            else:
                self.register_buffer("frame_p", P)
                self.register_buffer("frame_q", Q)
            # beta init 0: exact baseline at init, live gradient (see header)
            self.beta = nn.Parameter(torch.zeros(n_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.d_h).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.h, self.d_h).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.h, self.d_h).transpose(1, 2)

        cos, sin = rope_cache(T, self.d_h, x.device)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        logits = q @ k.transpose(-2, -1) / math.sqrt(self.d_h)  # (B,H,T,T)

        if self.gate:
            # a_i = <q_i, p>, c_j = <k_j, q_frame>; gate_ij = |a_i + c_j|
            a = torch.einsum("bhtd,hd->bht", q, self.frame_p)
            c = torch.einsum("bhtd,hd->bht", k, self.frame_q)
            gate = (a.unsqueeze(-1) + c.unsqueeze(-2)).abs()     # (B,H,T,T)
            logits = logits - self.beta.view(1, -1, 1, 1) * gate

        mask = torch.triu(torch.ones(T, T, dtype=torch.bool,
                                     device=x.device), diagonal=1)
        logits = logits.masked_fill(mask, float("-inf"))
        att = F.softmax(logits, dim=-1)
        if self.dropout > 0:
            att = F.dropout(att, self.dropout, self.training)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class Block(nn.Module):
    """Minimal pre-LN transformer block for Phase 2 wiring."""
    def __init__(self, d_model: int, n_heads: int, **attn_kwargs):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = ZDGatedCausalSelfAttention(d_model, n_heads, **attn_kwargs)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


# ----------------------------------------------------------------------
# Embedded test suite — run on a machine with torch:  python zda_layers.py
# ----------------------------------------------------------------------

def _tests():
    torch.manual_seed(0)
    dev = "cpu"

    # [T1] structure sanity: verified ZD pair annihilates under this build
    T64 = structure_tensor()
    (i, s1, j), (k, s2, l) = ZD_PAIR
    P = torch.zeros(16, dtype=torch.float64); P[i], P[j] = 1.0, s1
    Q = torch.zeros(16, dtype=torch.float64); Q[k], Q[l] = 1.0, s2
    prod = torch.einsum("kij,i,j->k", T64, P, Q)
    assert prod.abs().max() < 1e-12, "ZD pair failed to annihilate"
    print("[T1] ZD pair annihilates under torch build            OK")

    # [T2] PHMLinear: param count and W equals explicit kron sum
    lin = PHMLinear(64, 32, bias=False)
    assert lin.S.numel() == 64 * 32 // 16
    W = lin.weight()
    L = lin.L
    W_ref = torch.zeros_like(W)
    for kk in range(16):
        W_ref += torch.kron(L[kk], lin.S[kk])
    assert (W - W_ref).abs().max() < 1e-5
    print("[T2] PHMLinear params = dense/16; W == kron sum       OK")

    # [T3] gradcheck through PHMLinear (double precision)
    lin64 = PHMLinear(32, 32, bias=False).double()
    x = torch.randn(3, 32, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: lin64(t), (x,), eps=1e-6)
    print("[T3] gradcheck PHMLinear                              OK")

    # [T4] V2(beta=0) == baseline, exactly
    kw = dict(d_model=128, n_heads=4, proj="dense", seed=7)
    torch.manual_seed(7); m_gate = ZDGatedCausalSelfAttention(gate=True, **kw)
    torch.manual_seed(7); m_base = ZDGatedCausalSelfAttention(gate=False, **kw)
    m_gate.load_state_dict(m_base.state_dict(), strict=False)
    x = torch.randn(2, 10, 128)
    diff = (m_gate(x) - m_base(x)).abs().max().item()
    assert diff < 1e-6, f"beta=0 equality violated: {diff}"
    print(f"[T4] V2(beta=0) == baseline (max diff {diff:.1e})      OK")

    # [T5] gradient reaches beta at init (gate can turn on)
    y = m_gate(x).pow(2).sum(); y.backward()
    gnorm = m_gate.beta.grad.abs().sum().item()
    assert gnorm > 0, "beta gradient is dead at init"
    print(f"[T5] d(loss)/d(beta) nonzero at init ({gnorm:.3e})     OK")

    # [T6] causality: perturbing token t must not change outputs < t
    m = ZDGatedCausalSelfAttention(128, 4, gate=True)
    x2 = x.clone(); x2[:, 5:, :] += 1.0
    d = (m(x) - m(x2)).abs().amax(dim=(0, 2))
    assert d[:5].max() < 1e-6, "causal mask leak"
    print("[T6] causal mask holds under gate                     OK")

    # [T7] all six variant constructors build and run
    gen_kw = dict(d_model=64, n_heads=4)
    variants = {
        "B0/B1": dict(gate=False, proj="dense"),
        "V1":    dict(gate=False, proj="phm16"),
        "V2":    dict(gate=True, frame="zd", proj="dense"),
        "V3":    dict(gate=True, frame="random", proj="dense"),
        "V4":    dict(gate=False, proj="phm16_shuffled"),
    }
    xt = torch.randn(2, 8, 64)
    for name, v in variants.items():
        out = ZDGatedCausalSelfAttention(**gen_kw, **v)(xt)
        assert out.shape == xt.shape
    print("[T7] all variant constructors forward cleanly        OK")

    print("\nAll Phase 1 tests passed.")


if __name__ == "__main__":
    _tests()
