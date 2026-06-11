"""Flash Attention in JAX — companion code for Chapter 7.

Three implementations of the same function, in increasing order of control:
  1. attention_naive        — textbook reference (materializes the N x N matrix)
  2. flash_attention        — blockwise lax.scan forward + custom_vjp backward,
                              the FA-2 *algorithm* expressed in pure JAX
  3. flash_attention_pallas — a real fused GPU kernel via Pallas
                              (runs anywhere with interpret=True)

Run the self-tests:  python flash_attention_jax.py
All functions take one (N, d) head; lift to (batch, heads) with jax.vmap.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


# ---------------------------------------------------------------------------
# 1. Naive reference
# ---------------------------------------------------------------------------
def attention_naive(q, k, v, causal=False):
    N, d = q.shape
    s = (q @ k.T) * (d ** -0.5)
    if causal:
        s = jnp.where(jnp.tril(jnp.ones((N, N), bool)), s, -jnp.inf)
    return jax.nn.softmax(s, axis=-1) @ v


# ---------------------------------------------------------------------------
# 2a. Blockwise forward: FA-2's loop as a lax.scan over K/V tiles.
#     Returns (output, log-sum-exp) — exactly what a fused kernel writes.
# ---------------------------------------------------------------------------
def flash_forward_scan(q, k, v, block_k=128, causal=False):
    N, d = q.shape
    assert N % block_k == 0, "keep tiles even for clarity"
    scale = d ** -0.5
    Tc = N // block_k
    kb = k.reshape(Tc, block_k, d)
    vb = v.reshape(Tc, block_k, d)
    row_ids = jnp.arange(N)

    def body(carry, blk):
        m, l, acc = carry
        j, kj, vj = blk
        s = (q @ kj.T) * scale                       # (N, block_k) score tile
        if causal:
            col_ids = j * block_k + jnp.arange(block_k)
            s = jnp.where(col_ids[None, :] > row_ids[:, None], -jnp.inf, s)
        m_new = jnp.maximum(m, s.max(axis=1))
        alpha = jnp.exp(m - m_new)                   # repair factor for history
        p = jnp.exp(s - m_new[:, None])
        l = l * alpha + p.sum(axis=1)
        acc = acc * alpha[:, None] + p @ vj
        return (m_new, l, acc), None

    init = (jnp.full(N, -jnp.inf), jnp.zeros(N), jnp.zeros((N, d)))
    (m, l, acc), _ = jax.lax.scan(body, init, (jnp.arange(Tc), kb, vb))
    return acc / l[:, None], m + jnp.log(l)


# ---------------------------------------------------------------------------
# 2b. custom_vjp: forward saves (q, k, v, o, lse); backward recomputes P
#     tile by tile — the FA-2 backward, column-block "parallel" via scan.
# ---------------------------------------------------------------------------
@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def flash_attention(q, k, v, block_k=128, causal=False):
    o, _ = flash_forward_scan(q, k, v, block_k, causal)
    return o


def _flash_fwd(q, k, v, block_k, causal):
    o, lse = flash_forward_scan(q, k, v, block_k, causal)
    return o, (q, k, v, o, lse)          # residuals: O(N d) + O(N), never O(N^2)


def _flash_bwd(block_k, causal, res, do):
    q, k, v, o, lse = res
    N, d = q.shape
    scale = d ** -0.5
    D = jnp.sum(do * o, axis=1)          # the "delta" trick: rowsum(dO o O)
    Tc = N // block_k
    kb = k.reshape(Tc, block_k, d)
    vb = v.reshape(Tc, block_k, d)
    row_ids = jnp.arange(N)

    def body(dq, blk):
        j, kj, vj = blk
        s = (q @ kj.T) * scale
        if causal:
            col_ids = j * block_k + jnp.arange(block_k)
            s = jnp.where(col_ids[None, :] > row_ids[:, None], -jnp.inf, s)
        p = jnp.exp(s - lse[:, None])    # exact P tile, recomputed from lse
        dv_j = p.T @ do                  # local to this K/V tile
        dp = do @ vj.T
        ds = p * (dp - D[:, None]) * scale
        dk_j = ds.T @ q                  # local to this K/V tile
        dq = dq + ds @ kj                # accumulated across tiles (the
        return dq, (dk_j, dv_j)          # CUDA kernel's atomicAdd, as a carry)

    dq, (dk, dv) = jax.lax.scan(body, jnp.zeros_like(q), (jnp.arange(Tc), kb, vb))
    return dq, dk.reshape(N, d), dv.reshape(N, d)


flash_attention.defvjp(_flash_fwd, _flash_bwd)


# ---------------------------------------------------------------------------
# 3. Pallas kernel. One grid program per query block; K/V sliced on demand.
# ---------------------------------------------------------------------------
def _pallas_fwd_kernel(q_ref, k_ref, v_ref, o_ref, lse_ref, *,
                       block_k, scale, causal):
    block_q, d = q_ref.shape
    kv_len = k_ref.shape[0]
    i = pl.program_id(0)
    q = q_ref[...]                                    # (block_q, d)

    if causal:                                        # skip fully masked tiles
        num_tiles = ((i + 1) * block_q + block_k - 1) // block_k
    else:
        num_tiles = kv_len // block_k

    def body(j, carry):
        m, l, acc = carry
        kj = k_ref[pl.dslice(j * block_k, block_k), :]
        vj = v_ref[pl.dslice(j * block_k, block_k), :]
        s = jnp.dot(q, kj.T) * scale                  # (block_q, block_k)
        if causal:
            rows = i * block_q + jax.lax.broadcasted_iota(
                jnp.int32, (block_q, block_k), 0)
            cols = j * block_k + jax.lax.broadcasted_iota(
                jnp.int32, (block_q, block_k), 1)
            s = jnp.where(cols > rows, -jnp.inf, s)
        m_new = jnp.maximum(m, jnp.max(s, axis=1))
        alpha = jnp.exp(m - m_new)
        p = jnp.exp(s - m_new[:, None])
        l = l * alpha + jnp.sum(p, axis=1)
        acc = acc * alpha[:, None] + jnp.dot(p, vj)
        return m_new, l, acc

    init = (jnp.full((block_q,), -jnp.inf, jnp.float32),
            jnp.zeros((block_q,), jnp.float32),
            jnp.zeros((block_q, d), jnp.float32))
    m, l, acc = jax.lax.fori_loop(0, num_tiles, body, init)

    o_ref[...] = (acc / l[:, None]).astype(o_ref.dtype)
    lse_ref[...] = m + jnp.log(l)


@partial(jax.jit, static_argnames=("block_q", "block_k", "causal", "interpret"))
def flash_attention_pallas(q, k, v, block_q=64, block_k=64,
                           causal=False, interpret=False):
    N, d = q.shape
    assert N % block_q == 0 and N % block_k == 0
    kernel = partial(_pallas_fwd_kernel,
                     block_k=block_k, scale=d ** -0.5, causal=causal)
    o, lse = pl.pallas_call(
        kernel,
        grid=(N // block_q,),
        in_specs=[
            pl.BlockSpec((block_q, d), lambda i: (i, 0)),  # my Q tile
            pl.BlockSpec((N, d), lambda i: (0, 0)),        # all of K, sliced inside
            pl.BlockSpec((N, d), lambda i: (0, 0)),        # all of V, sliced inside
        ],
        out_specs=[
            pl.BlockSpec((block_q, d), lambda i: (i, 0)),  # my O tile
            pl.BlockSpec((block_q,), lambda i: (i,)),      # my lse slice
        ],
        out_shape=[
            jax.ShapeDtypeStruct((N, d), q.dtype),
            jax.ShapeDtypeStruct((N,), jnp.float32),
        ],
        interpret=interpret,
    )(q, k, v)
    return o, lse


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    on_gpu = jax.devices()[0].platform == "gpu"
    key = jax.random.key(0)
    kq, kk, kv, kd = jax.random.split(key, 4)
    N, d = 256, 64
    q = jax.random.normal(kq, (N, d), jnp.float32)
    k = jax.random.normal(kk, (N, d), jnp.float32)
    v = jax.random.normal(kv, (N, d), jnp.float32)
    do = jax.random.normal(kd, (N, d), jnp.float32)

    all_ok = True

    def check(name, a, b, tol=2e-5):
        global all_ok
        err = float(jnp.max(jnp.abs(a - b)))
        ok = err < tol
        all_ok &= ok
        print(f"{name:<42s} max err {err:.2e}  {'PASS' if ok else 'FAIL'}")

    for causal in (False, True):
        tag = "causal" if causal else "non-causal"
        ref = attention_naive(q, k, v, causal)

        o_scan, lse = flash_forward_scan(q, k, v, 64, causal)
        check(f"scan forward vs naive   [{tag}]", o_scan, ref)
        s = (q @ k.T) * (d ** -0.5)
        if causal:
            s = jnp.where(jnp.tril(jnp.ones((N, N), bool)), s, -jnp.inf)
        check(f"scan lse vs logsumexp   [{tag}]", lse,
              jax.scipy.special.logsumexp(s, axis=1))

        # gradients: custom_vjp vs autodiff through the naive version
        loss_flash = lambda q, k, v: jnp.sum(flash_attention(q, k, v, 64, causal) * do)
        loss_naive = lambda q, k, v: jnp.sum(attention_naive(q, k, v, causal) * do)
        g_flash = jax.grad(loss_flash, argnums=(0, 1, 2))(q, k, v)
        g_naive = jax.grad(loss_naive, argnums=(0, 1, 2))(q, k, v)
        for name, gf, gn in zip("dq dk dv".split(), g_flash, g_naive):
            check(f"custom_vjp {name} vs autodiff [{tag}]", gf, gn, tol=1e-4)

        o_pl, lse_pl = flash_attention_pallas(q, k, v, causal=causal,
                                              interpret=not on_gpu)
        check(f"pallas forward vs naive [{tag}]", o_pl, ref)
        check(f"pallas lse vs scan      [{tag}]", lse_pl, lse)

    # batched: vmap over (batch, head) just works
    qb = jax.random.normal(kq, (2, 4, N, d))
    kb = jax.random.normal(kk, (2, 4, N, d))
    vb = jax.random.normal(kv, (2, 4, N, d))
    mha = jax.vmap(jax.vmap(lambda q, k, v: flash_attention(q, k, v, 64, True)))
    ref = jax.vmap(jax.vmap(lambda q, k, v: attention_naive(q, k, v, True)))
    check("vmap batch x heads      [causal]", mha(qb, kb, vb), ref(qb, kb, vb))

    print("\nALL PASS" if all_ok else "\nSOME TESTS FAILED")
    raise SystemExit(0 if all_ok else 1)
