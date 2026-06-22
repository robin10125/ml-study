"""Deterministic checks tying transformer.py to the course's PyTorch.

Each check pins down one fact the course teaches — a shape, a number, an
autograd behaviour. Run:

    python test_torch_core.py     (~5 s on CPU; ends with ALL 7 CHECKS PASS)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import GPT, CausalSelfAttention, LayerNorm

torch.manual_seed(0)
ok_count = 0


def check(name, cond):
    global ok_count
    assert cond, f"FAILED: {name}"
    ok_count += 1
    print(f"  PASS  {name}")


# ----------------------------------------------------- 1. LayerNorm == nn.LayerNorm
# Our from-scratch LayerNorm must match PyTorch's, which means using the BIASED
# variance (unbiased=False). It also normalizes each token to mean 0, var 1.
x = torch.randn(2, 5, 16) * 3 + 7
ours = LayerNorm(16)
ref = nn.LayerNorm(16)                                # gamma=1, beta=0 by default
y = ours(x)
check("hand-built LayerNorm matches nn.LayerNorm and gives per-token mean 0, var 1",
      torch.allclose(y, ref(x), atol=1e-5)
      and torch.allclose(y.mean(-1), torch.zeros(2, 5), atol=1e-5)
      and torch.allclose(y.var(-1, unbiased=False), torch.ones(2, 5), atol=1e-3))

# ------------------------------------------------- 2. attention vs manual loop
attn = CausalSelfAttention(d_model=32, n_heads=4).eval()
H, hd, T = attn.n_heads, attn.d_head, 6
xx = torch.randn(1, T, 32)
with torch.no_grad():
    q = attn.wq(xx).view(1, T, H, hd)                # keep heads on axis 2 for the loop
    k = attn.wk(xx).view(1, T, H, hd)
    v = attn.wv(xx).view(1, T, H, hd)
    heads = []
    for h in range(H):                               # one head at a time, plain Python
        s = (q[0, :, h] @ k[0, :, h].T) / math.sqrt(hd)
        out_h = torch.zeros(T, hd)
        for i in range(T):
            w = torch.softmax(s[i, :i + 1], dim=-1)  # softmax over the visible prefix only
            out_h[i] = w @ v[0, :i + 1, h]
        heads.append(out_h)
    manual = attn.wo(torch.cat(heads, dim=-1))
    got = attn(xx)[0]
check("attention == per-head manual-loop reference",
      torch.allclose(got, manual, atol=1e-5))

# ------------------------------------------------------------- 3. causal mask
model = GPT(vocab=11, d_model=32, n_layers=2).eval()
toks = torch.randint(0, 11, (1, 8))
toks2 = toks.clone()
toks2[0, 5:] = (toks2[0, 5:] + 1) % 11                # corrupt the future (tokens >= 5)
with torch.no_grad():
    la, lb = model(toks), model(toks2)
check("causality: changing tokens >= t leaves logits < t unchanged",
      torch.allclose(la[0, :5], lb[0, :5], atol=1e-5))

# ----------------------------------------------------- 4. cross-entropy by hand
ce = F.cross_entropy(torch.tensor([[2.0, 0.0, 0.0]]), torch.tensor([0]))
check("F.cross_entropy of logits [2,0,0], target 0 = 0.2395 nats",
      abs(ce.item() - 0.2395) < 1e-3)

# --------------------------------------------------------- 5. ln(vocab) at init
V = 11
with torch.no_grad():
    logits = model(torch.randint(0, V, (16, 9)))
    loss0 = F.cross_entropy(logits.reshape(-1, V), torch.randint(0, V, (16 * 9,)))
check(f"untrained loss ~= ln(vocab) = {math.log(V):.3f}",
      abs(loss0.item() - math.log(V)) < 0.2)

# ----------------------------------- 6. autograd: backward fills & ACCUMULATES .grad
w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (w ** 2).sum()                                 # d/dw = 2w
loss.backward()
first = w.grad.clone()
loss2 = (w ** 2).sum()
loss2.backward()                                      # no zero_grad() -> grads ADD UP
check("backward() fills .grad with 2w and a second backward (no zero_grad) doubles it",
      torch.allclose(first, torch.tensor([2.0, 4.0, 6.0]))
      and torch.allclose(w.grad, 2 * first))

# ------------------------------------------------------ 7. Adam closed-form step
p = torch.tensor([2.0], requires_grad=True)
opt = torch.optim.Adam([p], lr=0.1)                   # plain Adam, no weight decay
p.grad = torch.tensor([0.5])                          # any positive gradient
opt.step()
# after one step m-hat = g and v-hat = g^2, so the move is exactly -lr * sign(g)
check("first Adam step moves the param by exactly lr * sign(g)",
      abs(p.item() - 1.9) < 1e-6)

print(f"\nALL {ok_count} CHECKS PASS")
