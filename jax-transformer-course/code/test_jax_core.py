"""Deterministic checks tying transformer.py to the course's math.

Run:  python test_jax_core.py     (~30 s on CPU; ends with ALL 7 CHECKS PASS)
"""

import jax
import jax.numpy as jnp
import numpy as np

from transformer import (N_HEADS, adam_init, adam_update, attention,
                         cross_entropy, forward, init_params, layer_norm,
                         loss_fn)

ok_count = 0


def check(name, cond):
    global ok_count
    assert cond, f"FAILED: {name}"
    ok_count += 1
    print(f"  PASS  {name}")


# ---------------------------------------------------------------- 1. layer_norm
x = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 16)) * 3 + 7
y = layer_norm(x, jnp.ones(16), jnp.zeros(16))
check("layer_norm gives per-token mean 0, var 1 (ch5 §norm)",
      np.allclose(y.mean(-1), 0, atol=1e-5) and np.allclose(y.var(-1), 1, atol=1e-3))

# ------------------------------------------------- 2. attention vs manual loop
key = jax.random.PRNGKey(1)
D, T, hd = 32, 6, 32 // N_HEADS
ks = jax.random.split(key, 5)
p = {w: jax.random.normal(ks[i], (D, D)) / np.sqrt(D)
     for i, w in enumerate(["wq", "wk", "wv", "wo"])}
xx = jax.random.normal(ks[4], (1, T, D))

q = (xx @ p["wq"]).reshape(1, T, N_HEADS, hd)
k = (xx @ p["wk"]).reshape(1, T, N_HEADS, hd)
v = (xx @ p["wv"]).reshape(1, T, N_HEADS, hd)
heads = []
for h in range(N_HEADS):                      # one head at a time, plain loops
    s = np.array(q[0, :, h] @ k[0, :, h].T) / np.sqrt(hd)
    out_h = np.zeros((T, hd))
    for i in range(T):
        w = np.exp(s[i, :i + 1] - s[i, :i + 1].max())
        w = w / w.sum()                       # softmax over visible prefix only
        out_h[i] = w @ np.array(v[0, :i + 1, h])
    heads.append(out_h)
manual = np.concatenate(heads, -1) @ np.array(p["wo"])
check("attention == per-head manual-loop reference (ch5 §attention, ch6 §attn-code)",
      np.allclose(np.array(attention(p, xx)[0]), manual, atol=1e-4))

# ------------------------------------------------------------- 3. causal mask
key = jax.random.PRNGKey(2)
params = init_params(key, vocab=11, d_model=32, n_layers=2)
toks = jax.random.randint(jax.random.PRNGKey(3), (1, 8), 0, 11)
toks2 = toks.at[0, 5:].set((toks[0, 5:] + 1) % 11)   # corrupt the future
la, lb = forward(params, toks), forward(params, toks2)
check("causality: changing tokens >= t leaves logits < t unchanged (ch5 §mask)",
      np.allclose(la[0, :5], lb[0, :5], atol=1e-5))

# ----------------------------------------------------- 4. cross-entropy by hand
ce = cross_entropy(jnp.array([[2.0, 0.0, 0.0]]), jnp.array([0]))
check("cross-entropy of logits [2,0,0], target 0 = 0.2395 nats (ch5 §loss)",
      abs(float(ce) - 0.2395) < 1e-3)

# --------------------------------------------------------- 5. ln(vocab) at init
V = 11
loss0 = loss_fn(params, jax.random.randint(jax.random.PRNGKey(4), (16, 9), 0, V))
check(f"untrained loss ~= ln(vocab) = {np.log(V):.3f} (ch5 §loss, ch7 §debug)",
      abs(float(loss0) - np.log(V)) < 0.15)

# ------------------------------------------------ 6. grad vs finite differences
def f(params):
    return loss_fn(params, toks.astype(jnp.int32).repeat(2, 0))

g = jax.grad(f)(params)["embed"]
eps = 1e-3
for (i, j) in [(3, 0), (7, 5)]:
    pp = lambda s: {**params, "embed": params["embed"].at[i, j].add(s)}
    fd = (f(pp(eps)) - f(pp(-eps))) / (2 * eps)
    assert abs(float(fd) - float(g[i, j])) < 1e-2, "grad mismatch"
check("jax.grad matches central finite differences on loss_fn (ch2 §grad)", True)

# ------------------------------------------------------ 7. Adam closed-form step
w = {"w": jnp.array(2.0)}
grads = {"w": jnp.array(0.5)}
state = adam_init(w)
w2, state = adam_update(w, grads, state, step=1, lr=0.1)
# after one step, m-hat = g and v-hat = g^2, so the update is lr * sign(g)
check("first Adam step moves by exactly lr * sign(g) (ch7 §adam)",
      abs(float(w2["w"]) - 1.9) < 1e-6)

print(f"\nALL {ok_count} CHECKS PASS")
