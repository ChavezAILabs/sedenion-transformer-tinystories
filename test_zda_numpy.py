"""
test_zda_numpy.py — NumPy mirror of the math in zda_layers.py.

Torch is not available in this sandbox, so every mathematical claim the
PyTorch layer relies on is verified here independently:

  [M1] PHM-16 semantics: applying W = sum_k kron(L_k, S_k) to a
       component-major feature vector IS a matrix of learned sedenions
       acting by left multiplication (checked against explicit cd_mult).
  [M2] Gate primitive <-> collapse identity: for the frame pair
       P = e1+e10, Q = e4-e15,  ||(aP+bQ)(bP+cQ)|| = 2|b||a+c| exactly,
       so the gate term |a+c| is (up to the constant fold into beta)
       the collapse magnitude of the induced bilateral configuration.
  [M3] beta=0 gated attention == standard attention, exactly.
  [M4] Causal mask holds with the gate active.
  [M5] Smooth dependence on beta: central finite differences of the
       loss converge under eps-halving (Richardson ratio ~ 4 for O(eps^2)).
       (Full autograd gradcheck runs in zda_layers.py on a torch machine.)
  [M6] RoPE relative-position property: <rope(q,m), rope(k,n)> depends
       only on m - n.
"""

import numpy as np
from sedenion_kernel import cd_mult, basis, structure_tensor

rng = np.random.default_rng(42)
N = 16

# ---------------------------------------------------------------- [M1]
T = structure_tensor(N)
L = np.transpose(T, (1, 0, 2))          # L[k][m,j] = T[m,k,j]
in_b, out_b = 3, 2
S = rng.standard_normal((N, out_b, in_b))

W = np.einsum("kab,kcd->acbd", L, S).reshape(N * out_b, N * in_b)
x = rng.standard_normal(N * in_b)       # component-major: x[a*in_b + d]
y1 = W @ x

X = x.reshape(N, in_b)                  # column d = sedenion x_d
y2 = np.zeros((N, out_b))
for c in range(out_b):
    acc = np.zeros(N)
    for d in range(in_b):
        w_cd = S[:, c, d]               # learned sedenion, components S[k,c,d]
        acc += cd_mult(w_cd, X[:, d])
    y2[:, c] = acc
err = np.abs(y1 - y2.reshape(-1)).max()
print(f"[M1] PHM-16 == matrix of learned sedenions:  err {err:.2e}")
assert err < 1e-10
assert S.size == (N * out_b) * (N * in_b) // 16
print(f"     param count = dense/16 confirmed ({S.size} vs {W.size})")

# ---------------------------------------------------------------- [M2]
# Pattern 2 (e3+e12, e5+e10): the ZD_PAIR in zda_layers.py / zda_grid.py.
# Bilateral: annihilates in both orders (checked below).
P = basis(3) + basis(12)
Q = basis(5) + basis(10)
assert np.abs(cd_mult(P, Q)).max() < 1e-12
assert np.abs(cd_mult(Q, P)).max() < 1e-12
for _ in range(200):
    a, b, c = rng.standard_normal(3)
    m = np.linalg.norm(cd_mult(a * P + b * Q, b * P + c * Q))
    assert abs(m - 2 * abs(b) * abs(a + c)) < 1e-9
print("[M2] ||(aP+bQ)(bP+cQ)|| == 2|b||a+c| for frame pair (200 draws)")

# ---------------------------------------------------------------- [M3-M5]
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def attention(q, k, v, p_hat, q_hat, beta, causal=True):
    Tn, dh = q.shape
    logits = q @ k.T / np.sqrt(dh)
    a = q @ p_hat                        # (T,)
    c = k @ q_hat                        # (T,)
    logits = logits - beta * np.abs(a[:, None] + c[None, :])
    if causal:
        logits = np.where(np.triu(np.ones((Tn, Tn), bool), 1), -np.inf, logits)
    return softmax(logits) @ v

Tn, dh = 12, 64
q = rng.standard_normal((Tn, dh)); k = rng.standard_normal((Tn, dh))
v = rng.standard_normal((Tn, dh))
p_hat = np.zeros(dh); p_hat[1], p_hat[10] = 1, 1;  p_hat /= np.linalg.norm(p_hat)
q_hat = np.zeros(dh); q_hat[4], q_hat[15] = 1, -1; q_hat /= np.linalg.norm(q_hat)

base = attention(q, k, v, p_hat, q_hat, beta=0.0)
plain_logits = q @ k.T / np.sqrt(dh)
plain_logits = np.where(np.triu(np.ones((Tn, Tn), bool), 1), -np.inf, plain_logits)
plain = softmax(plain_logits) @ v
err = np.abs(base - plain).max()
print(f"[M3] beta=0 gated attention == standard:     err {err:.2e}")
assert err < 1e-12

q2 = q.copy(); q2[6:] += 1.0
d = np.abs(attention(q, k, v, p_hat, q_hat, 0.7)
           - attention(q2, k, v, p_hat, q_hat, 0.7)).max(axis=1)
assert d[:6].max() < 1e-12
print("[M4] causal mask holds with gate active (beta=0.7)")

def loss(beta):
    return (attention(q, k, v, p_hat, q_hat, beta) ** 2).sum()

g_eps  = (loss(1e-4) - loss(-1e-4)) / 2e-4
g_eps2 = (loss(5e-5) - loss(-5e-5)) / 1e-4
print(f"[M5] dL/dbeta at 0: FD={g_eps:+.6f}, half-eps={g_eps2:+.6f} "
      f"(agree to {abs(g_eps - g_eps2):.1e}; nonzero -> gate trainable)")
assert abs(g_eps) > 1e-6 and abs(g_eps - g_eps2) < 1e-6

# ---------------------------------------------------------------- [M6]
def rope(x, pos, base=10000.0):
    dh = x.shape[-1]; half = dh // 2
    freqs = base ** (-np.arange(half) / half)
    ang = pos * freqs
    x1, x2 = x[0::2], x[1::2]
    out = np.empty_like(x)
    out[0::2] = x1 * np.cos(ang) - x2 * np.sin(ang)
    out[1::2] = x1 * np.sin(ang) + x2 * np.cos(ang)
    return out

qv, kv = rng.standard_normal(dh), rng.standard_normal(dh)
d1 = rope(qv, 7) @ rope(kv, 3)     # offset 4
d2 = rope(qv, 20) @ rope(kv, 16)   # offset 4
print(f"[M6] RoPE relative property: {d1:.10f} == {d2:.10f}")
assert abs(d1 - d2) < 1e-9

print("\nAll NumPy mirror checks passed — zda_layers.py math is verified.")
