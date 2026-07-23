"""phase4_layers.py — PyTorch K3 norm-ratio attention (Phase 4 primary
kernel, ZDA_phase4_spec.md SS2), ported from the validated NumPy mirror
`test_phase4_numpy.py`. Run this file to execute tests P4T1-P4T7.

Mechanism, per head h:
  q_i, k_j = 16-component sedenion coordinates from learned projections
  positional: R_8(w_h * pos) in-algebra rotation (unique compatible
              generator; per-head single frequency w_h — both facts
              derived and verified in phase4_positional.py / memo SS4).
              Ladder FROZEN at spec v1.0 (SS2): w_h = (2pi/1024)^(h/(H-1)),
              i.e. geometric from w_0 = 1 down to w_{H-1} = 2pi/1024;
              exposed as the buffer `omega_h` (length H), which
              phase4_grid.py preflight [P4] locates and verifies.
  r2_ij = |q_i (x) k_j|^2 / (|q_i|^2 |k_j|^2 + eps_div)
  score = -gamma_h * r2_ij       (mandatory: no parameter setting
                                  recovers dot-product attention)
  causal mask -> softmax -> @ v (v/out projections are standard dense).

Design rules carried from the spec:
  * MANDATORY kernel: there is deliberately no dot-product term and no
    beta=0-style escape to dense attention. The T4 anchor is replaced by
    P4T1 (structural nulls), P4T2 (K1-degeneracy guard) and P4T3
    (octonion-degeneracy end-to-end).
  * fp32 score path (spec SS9.2 hygiene rule): q/k projections outputs,
    the sedenion product, r2 and the score are computed in float32
    regardless of input dtype; fp16/bf16 storage of q,k quantizes
    manifold distance to ~1e-3 and kills the r2 < 1e-4 diagnostic bin.
  * gamma is per-head, init positive, NOT weight-decayed (enforce in the
    optimizer builder as for beta in Phase 3); sign-flipped g is a
    reparameterization, never a separate variant.
  * eps_div = 1e-12 guards only against zero-norm projections; it is far
    below every diagnostic bin.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from sedenion_kernel import basis, structure_tensor

N_ALG = 16
L_MAX = 1024   # length-gen anchor pinning the frozen ladder (spec SS2 v1.0)


def frozen_omega_ladder(n_heads: int, l_max: int = L_MAX) -> list[float]:
    """Frozen spec-SS2 v1.0 per-head frequency ladder: geometric with
    task-pinned endpoints, omega_0 = 1 down to omega_{H-1} = 2*pi/l_max.
    Identical for S/X (R_8 action) and D0p/D1 (per-head single-frequency
    RoPE) — the positional-matching invariant. Same formula as
    phase4_grid.omega_ladder; verified there by preflight [P4]."""
    return [(2 * math.pi / l_max) ** (h / max(n_heads - 1, 1))
            for h in range(n_heads)]


def shuffled_structure_tensor(seed: int) -> np.ndarray:
    """Variant X: pinned recipe from phase4_shuffledT_nulls.py
    (numpy port of zda_layers.shuffled_left_mult_matrices)."""
    g = np.random.default_rng(seed)
    L = np.zeros((16, 16, 16))
    L[0] = np.eye(16)
    for k in range(1, 16):
        perm = g.permutation(16)
        signs = g.integers(0, 2, 16) * 2 - 1
        L[k, np.arange(16), perm] = signs
    return np.transpose(L, (1, 0, 2))


class K3Attention(nn.Module):
    """Norm-ratio sedenion attention. `tensor` selects S (None -> true
    T16), X (pass shuffled_structure_tensor(seed)) or the octonion
    degeneracy probe (structure_tensor(8) with n_alg=8)."""

    def __init__(self, d_model: int, n_heads: int, n_alg: int = N_ALG,
                 causal: bool = True, gamma_init: float = 1.0,
                 omega_h=None, tensor: np.ndarray | None = None):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.a, self.causal = n_heads, n_alg, causal
        self.dv = d_model // n_heads
        T = structure_tensor(n_alg) if tensor is None else tensor
        self.register_buffer("T", torch.tensor(T, dtype=torch.float32))
        # positional generator: the algebra's doubling index (8 for
        # sedenions — unique compatible generator; n_alg//2 generally)
        self.register_buffer("Lrot", self.T[:, n_alg // 2, :].clone())
        # frozen per-head ladder (spec SS2 v1.0); the attribute NAME is
        # load-bearing — phase4_grid preflight [P4] looks for `omega_h`
        if omega_h is None:
            omega_h = frozen_omega_ladder(n_heads)
        self.register_buffer("omega_h", torch.tensor(omega_h,
                                                     dtype=torch.float32))
        self.wq = nn.Linear(d_model, n_heads * n_alg, bias=False)
        self.wk = nn.Linear(d_model, n_heads * n_alg, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.gamma = nn.Parameter(torch.full((n_heads,), gamma_init))
        self.eps_div = 1e-12

    def _rotate(self, u, pos):
        # u: (B,H,T,A); pos: (T,) -> R(omega_h * pos) u, in fp32
        th = self.omega_h.view(1, -1, 1, 1) * pos.view(1, 1, -1, 1)
        return torch.cos(th) * u + torch.sin(th) * (u @ self.Lrot.T)

    def scores(self, x, pos_offset: float = 0.0):
        """r2 and masked score, fp32 end-to-end (returns (r2, score))."""
        B, T, _ = x.shape
        xf = x.float()
        q = self.wq(xf).view(B, T, self.h, self.a).transpose(1, 2)
        k = self.wk(xf).view(B, T, self.h, self.a).transpose(1, 2)
        pos = torch.arange(T, dtype=torch.float32,
                           device=x.device) + pos_offset
        q, k = self._rotate(q, pos), self._rotate(k, pos)
        # two-step contraction (same math as the mij triple einsum, ~8x
        # cheaper): K_rot[b,h,s,i,m] = sum_j T[m,i,j] k_j, then contract i
        k_rot = torch.einsum("mij,bhsj->bhsim", self.T, k)
        prod = torch.einsum("bhti,bhsim->bhtsm", q, k_rot)
        num = prod.pow(2).sum(-1)
        den = (q.pow(2).sum(-1).unsqueeze(-1)
               * k.pow(2).sum(-1).unsqueeze(-2) + self.eps_div)
        r2 = num / den
        s = -self.gamma.view(1, -1, 1, 1) * r2
        if self.causal:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool,
                                         device=x.device), 1)
            s = s.masked_fill(mask, float("-inf"))
        return r2, s

    def forward(self, x, pos_offset: float = 0.0):
        B, T, _ = x.shape
        _, s = self.scores(x, pos_offset)
        att = torch.softmax(s, dim=-1)
        v = self.wv(x.float()).view(B, T, self.h, self.dv).transpose(1, 2)
        y = (att @ v).transpose(1, 2).reshape(B, T, -1)
        return self.wo(y).to(x.dtype)


# ----------------------------------------------------------------------
# Tests P4T1-P4T7 (mirror: test_phase4_numpy.py)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, d, H = 2, 12, 48, 3

    # [P4T1] structural nulls survive the torch build exactly
    m = K3Attention(d, H)
    blades = []
    for i in range(1, 16):
        for j in range(i + 1, 16):
            for sgn in (1.0, -1.0):
                v = torch.zeros(16); v[i], v[j] = 1.0, sgn
                blades.append(v)
    Bt = torch.stack(blades)
    prods = torch.einsum("mij,ai,bj->abm", m.T, Bt, Bt)
    nulls = (prods.abs().amax(-1) == 0.0)
    assert nulls.sum().item() == 336, nulls.sum()
    print("[P4T1] 336 structural nulls exact under torch      OK")

    # [P4T2] K1-degeneracy guard: separable probe from the SAME q,k is
    # query-independent; the K3 score is not.
    x = torch.randn(B, T, d)
    xf = x.float()
    q = m.wq(xf).view(B, T, H, 16).transpose(1, 2)
    k = m.wk(xf).view(B, T, H, 16).transpose(1, 2)
    p_hat = torch.tensor(basis(3) + basis(12), dtype=torch.float32)
    q_hat = torch.tensor(basis(5) + basis(10), dtype=torch.float32)
    p_hat, q_hat = p_hat / p_hat.norm(), q_hat / q_hat.norm()
    s_k1 = (q @ p_hat).unsqueeze(-1) + (k @ q_hat).unsqueeze(-2)
    rows = torch.softmax(s_k1, -1)
    spread_k1 = (rows - rows[..., :1, :]).abs().max().item()
    r2, _ = m.scores(x)
    rows3 = torch.softmax(-r2, -1)
    spread_k3 = (rows3 - rows3[..., :1, :]).abs().max().item()
    assert spread_k1 < 1e-6 and spread_k3 > 1e-3
    print(f"[P4T2] K1 probe degenerate ({spread_k1:.1e}), "
          f"K3 query-dependent ({spread_k3:.3f})   OK")

    # [P4T3] octonion swap -> attention provably query-independent
    m8 = K3Attention(d, H, n_alg=8, causal=False)
    r2o, _ = m8.scores(x)
    rows8 = torch.softmax(-r2o, -1)
    spread8 = (rows8 - rows8[..., :1, :]).abs().max().item()
    assert (r2o - 1).abs().max().item() < 1e-5 and spread8 < 1e-6
    print(f"[P4T3] octonion K3 degenerate end-to-end "
          f"(|r2-1| {(r2o-1).abs().max():.1e})     OK")

    # [P4T4] causality
    x2 = x.clone(); x2[:, 7:] += 1.0
    d47 = (m(x)[:, :7] - m(x2)[:, :7]).abs().max().item()
    assert d47 < 1e-6, d47
    print(f"[P4T4] causal mask holds under K3 score ({d47:.1e})   OK")

    # [P4T5] position-shift invariance (R_8 relative property, per-head
    # frequencies, end-to-end through the full layer)
    dshift = (m(x) - m(x, pos_offset=23.0)).abs().max().item()
    assert dshift < 1e-4, dshift
    print(f"[P4T5] output invariant to global pos shift ({dshift:.1e}) OK")

    # [P4T6] gradients reach every parameter; gamma grad nonzero
    m.zero_grad()
    m(x).pow(2).mean().backward()
    for n_, p_ in m.named_parameters():
        assert p_.grad is not None and torch.isfinite(p_.grad).all(), n_
    g = m.gamma.grad.abs().sum().item()
    assert g > 0
    print(f"[P4T6] grads finite everywhere, d(loss)/d(gamma) {g:.2e}  OK")

    # [P4T7] NumPy mirror equality on the layer's own weights (head 0)
    import numpy as _np
    T16 = structure_tensor(16)
    L8n = T16[:, 8, :]
    Wq = m.wq.weight.detach().numpy().reshape(H, 16, d)[0]
    Wk = m.wk.weight.detach().numpy().reshape(H, 16, d)[0]
    xn = x[0].numpy().astype(_np.float64)
    qn, kn = xn @ Wq.T, xn @ Wk.T
    w0 = float(m.omega_h[0])
    pos = _np.arange(T, dtype=float)
    def rotn(u):
        th = w0 * pos
        return (_np.cos(th)[:, None] * u
                + _np.sin(th)[:, None] * (u @ L8n.T))
    qn, kn = rotn(qn), rotn(kn)
    pr = _np.einsum("mij,ti,sj->tsm", T16, qn, kn)
    r2n = (pr ** 2).sum(-1) / ((qn ** 2).sum(-1)[:, None]
                               * (kn ** 2).sum(-1)[None, :] + 1e-12)
    r2t = m.scores(x)[0][0, 0].detach().numpy()
    dmirror = float(_np.abs(r2n - r2t).max())
    assert dmirror < 1e-4, dmirror
    print(f"[P4T7] torch r2 == NumPy mirror r2 ({dmirror:.1e})      OK")

    # variant constructors forward cleanly (S, X seeds, octonion probe)
    for tns in (None, shuffled_structure_tensor(1337),
                shuffled_structure_tensor(1338),
                shuffled_structure_tensor(1339)):
        K3Attention(d, H, tensor=tns)(x)
    print("[P4T8] S + X(1337/1338/1339) constructors forward   OK")

    print("\nAll Phase 4 torch layer tests passed.")
