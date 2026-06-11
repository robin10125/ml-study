"""A character-level transformer language model in pure JAX.

No Flax, no Optax, no hidden state anywhere: the model is a function
`forward(params, tokens) -> logits`, the parameters are a nested dict of
arrays (a pytree), the optimizer is two more pytrees, and one jitted
`train_step` advances everything.

This is the reference implementation for the course in jax-transformer-course/.
Every function here is one the course asks you to write from memory.

Run:  python transformer.py            (~2 min on CPU; prints PASS at the end)
      python transformer.py --steps 200   (quick smoke test, will not PASS)
"""

import argparse
import time

import jax
import jax.numpy as jnp

# ----------------------------------------------------------------------------
# Data: a tiny public-domain corpus (opening of Alice in Wonderland).
# Small on purpose — a few hundred thousand parameters can *memorize* it,
# so success is unambiguous: the loss collapses and samples quote the text.
# ----------------------------------------------------------------------------

CORPUS = """Alice was beginning to get very tired of sitting by her sister on the
bank, and of having nothing to do: once or twice she had peeped into the
book her sister was reading, but it had no pictures or conversations in
it, "and what is the use of a book," thought Alice, "without pictures or
conversations?"

So she was considering in her own mind (as well as she could, for the
hot day made her feel very sleepy and stupid), whether the pleasure of
making a daisy-chain would be worth the trouble of getting up and
picking the daisies, when suddenly a White Rabbit with pink eyes ran
close by her.

There was nothing so very remarkable in that; nor did Alice think it so
very much out of the way to hear the Rabbit say to itself, "Oh dear! Oh
dear! I shall be late!" (when she thought it over afterwards, it
occurred to her that she ought to have wondered at this, but at the time
it all seemed quite natural); but when the Rabbit actually took a watch
out of its waistcoat-pocket, and looked at it, and then hurried on,
Alice started to her feet, for it flashed across her mind that she had
never before seen a rabbit with either a waistcoat-pocket, or a watch to
take out of it, and burning with curiosity, she ran across the field
after it, and fortunately was just in time to see it pop down a large
rabbit-hole under the hedge.

In another moment down went Alice after it, never once considering how
in the world she was to get out again. The rabbit-hole went straight on
like a tunnel for some way, and then dipped suddenly down, so suddenly
that Alice had not a moment to think about stopping herself before she
found herself falling down a very deep well.

Either the well was very deep, or she fell very slowly, for she had
plenty of time as she went down to look about her and to wonder what was
going to happen next. First, she tried to look down and make out what
she was coming to, but it was too dark to see anything; then she looked
at the sides of the well, and noticed that they were filled with
cupboards and book-shelves; here and there she saw maps and pictures
hung upon pegs.
"""

# ----------------------------------------------------------------------------
# Model. Decoder-only, pre-LN, learned positional embeddings, tied unembedding.
# Shapes use B = batch, T = sequence length, D = d_model, H = heads.
# ----------------------------------------------------------------------------


N_HEADS = 4  # static config, deliberately NOT inside the params pytree:
             # every leaf of params must be a trainable array (grad/Adam map
             # over all leaves), so plain Python ints don't belong there.


def layer_norm(x, gamma, beta, eps=1e-5):
    """Normalize the last axis to mean 0 / variance 1, then scale and shift."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps) * gamma + beta


def attention(p, x):
    """Causal multi-head self-attention. x: (B, T, D) -> (B, T, D)."""
    B, T, D = x.shape
    H = N_HEADS
    hd = D // H

    q = (x @ p["wq"]).reshape(B, T, H, hd).transpose(0, 2, 1, 3)  # (B, H, T, hd)
    k = (x @ p["wk"]).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
    v = (x @ p["wv"]).reshape(B, T, H, hd).transpose(0, 2, 1, 3)

    scores = q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(hd)           # (B, H, T, T)
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    scores = jnp.where(mask, scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)

    out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    return out @ p["wo"]


def mlp(p, x):
    """Position-wise feed-forward: D -> d_ff -> D with GELU."""
    return jax.nn.gelu(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]


def block(p, x):
    """One pre-LN transformer block: x + attn(LN(x)), then x + mlp(LN(x))."""
    x = x + attention(p["attn"], layer_norm(x, p["ln1_g"], p["ln1_b"]))
    x = x + mlp(p["mlp"], layer_norm(x, p["ln2_g"], p["ln2_b"]))
    return x


def forward(params, tokens):
    """tokens: (B, T) ints -> logits: (B, T, vocab)."""
    B, T = tokens.shape
    x = params["embed"][tokens] + params["pos"][:T]               # (B, T, D)
    for p in params["blocks"]:
        x = block(p, x)
    x = layer_norm(x, params["lnf_g"], params["lnf_b"])
    return x @ params["embed"].T                                   # weight tying


def cross_entropy(logits, targets):
    """Mean -log p(target). logits: (..., vocab), targets: (...) ints."""
    logp = jax.nn.log_softmax(logits, axis=-1)
    picked = jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    return -picked.mean()


def loss_fn(params, tokens):
    """tokens: (B, T+1) ints. Predict token t+1 from tokens <= t."""
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    logits = forward(params, inputs)
    return cross_entropy(logits, targets)


def init_params(key, vocab, d_model=128, n_layers=3, d_ff=512, max_T=256):
    """Build the parameter pytree. Linear weights are scaled by 1/sqrt(fan_in)
    so activations keep roughly unit variance at depth; embeddings start small."""

    def dense(key, n_in, n_out):
        return jax.random.normal(key, (n_in, n_out)) / jnp.sqrt(n_in)

    key, ek, pk = jax.random.split(key, 3)
    blocks = []
    for _ in range(n_layers):
        key, kq, kk, kv, ko, k1, k2 = jax.random.split(key, 7)
        blocks.append({
            "ln1_g": jnp.ones(d_model), "ln1_b": jnp.zeros(d_model),
            "ln2_g": jnp.ones(d_model), "ln2_b": jnp.zeros(d_model),
            "attn": {
                "wq": dense(kq, d_model, d_model),
                "wk": dense(kk, d_model, d_model),
                "wv": dense(kv, d_model, d_model),
                "wo": dense(ko, d_model, d_model),
            },
            "mlp": {
                "w1": dense(k1, d_model, d_ff), "b1": jnp.zeros(d_ff),
                "w2": dense(k2, d_ff, d_model), "b2": jnp.zeros(d_model),
            },
        })
    return {
        "embed": jax.random.normal(ek, (vocab, d_model)) * 0.02,
        "pos": jax.random.normal(pk, (max_T, d_model)) * 0.02,
        "blocks": blocks,
        "lnf_g": jnp.ones(d_model), "lnf_b": jnp.zeros(d_model),
    }


# ----------------------------------------------------------------------------
# Adam, by hand. State = two pytrees of the same shape as params.
# ----------------------------------------------------------------------------


def adam_init(params):
    zeros = lambda p: jnp.zeros_like(p)
    return {"m": jax.tree.map(zeros, params), "v": jax.tree.map(zeros, params)}


def adam_update(params, grads, state, step, lr, b1=0.9, b2=0.999, eps=1e-8):
    """One Adam step. Returns (new_params, new_state). `step` is 1-based."""
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, state["v"], grads)
    mhat_scale = 1.0 / (1 - b1 ** step)
    vhat_scale = 1.0 / (1 - b2 ** step)
    params = jax.tree.map(
        lambda p, m, v: p - lr * (m * mhat_scale)
        / (jnp.sqrt(v * vhat_scale) + eps),
        params, m, v)
    return params, {"m": m, "v": v}


# ----------------------------------------------------------------------------
# Data pipeline and sampling
# ----------------------------------------------------------------------------


def get_batch(key, data, batch_size, T):
    """Sample batch_size random windows of length T+1 from the token array."""
    starts = jax.random.randint(key, (batch_size,), 0, data.shape[0] - T - 1)
    idx = starts[:, None] + jnp.arange(T + 1)[None, :]
    return data[idx]                                               # (B, T+1)


def sample(params, key, prompt_ids, n_new, T, temperature=0.8):
    """Autoregressive sampling: feed the (truncated) context, take the last
    position's logits, draw one token, append, repeat."""
    ids = list(prompt_ids)
    for _ in range(n_new):
        ctx = jnp.array(ids[-T:])[None, :]                        # (1, <=T)
        logits = forward(params, ctx)[0, -1]                      # (vocab,)
        key, sub = jax.random.split(key)
        ids.append(int(jax.random.categorical(sub, logits / temperature)))
    return ids


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # tokenizer: every distinct character is a token
    chars = sorted(set(CORPUS))
    vocab = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = jnp.array([stoi[c] for c in CORPUS])
    print(f"corpus: {data.shape[0]} chars, vocab = {vocab}")

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params = init_params(init_key, vocab)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"params: {n_params:,}  (expect first loss near ln {vocab} = "
          f"{jnp.log(vocab):.2f})")

    opt_state = adam_init(params)

    @jax.jit
    def train_step(params, opt_state, step, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        params, opt_state = adam_update(params, grads, opt_state, step, args.lr)
        return params, opt_state, loss

    t0 = time.time()
    recent = []
    for step in range(1, args.steps + 1):
        key, bkey = jax.random.split(key)
        batch = get_batch(bkey, data, args.batch_size, args.seq_len)
        params, opt_state, loss = train_step(params, opt_state, step, batch)
        recent.append(float(loss))
        if step % 200 == 0 or step == 1:
            avg = sum(recent[-100:]) / len(recent[-100:])
            print(f"step {step:5d}  loss {avg:.3f}  ({time.time()-t0:.0f}s)")

    final = sum(recent[-100:]) / len(recent[-100:])
    print(f"\nFINAL: average loss over last 100 steps = {final:.3f}")

    key, skey = jax.random.split(key)
    out = sample(params, skey, [stoi[c] for c in "Alice "], 250, args.seq_len)
    print("\nsample:\n" + "".join(chars[i] for i in out))

    if final < 1.0:
        print(f"\nPASS: model fits the corpus (loss {final:.3f} < 1.0; "
              f"untrained baseline ln {vocab} = {float(jnp.log(vocab)):.2f}).")
    else:
        print(f"\nFAIL: loss {final:.3f} did not reach 1.0.")


if __name__ == "__main__":
    main()
