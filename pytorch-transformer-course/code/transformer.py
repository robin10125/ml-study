"""A character-level transformer language model in idiomatic PyTorch.

This is the reference implementation for the course in
pytorch-transformer-course/. The whole point of the course is to be able to
write this file from an empty editor, so it is deliberately small and plain:
no tricks, every layer an ``nn.Module``, every shape commented.

The model is a decoder-only ("GPT-style") transformer: token + learned
positional embeddings, a stack of pre-LayerNorm blocks (causal multi-head
self-attention + an MLP, each wrapped in a residual connection), a final
LayerNorm, and a tied unembedding. Shapes throughout use
B = batch, T = sequence length, D = d_model, H = n_heads, hd = head dim.

Run:  python transformer.py            (~2-4 min on CPU; prints PASS at the end)
      python transformer.py --steps 200   (quick smoke test, will NOT PASS)
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Data: a tiny public-domain corpus (the opening of Alice in Wonderland).
# Small on purpose — a few hundred thousand parameters can *memorize* it, so
# success is unambiguous: the loss collapses far below the untrained baseline
# of ln(vocab), and samples start to quote the text back.
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
# LayerNorm, built from scratch. PyTorch ships nn.LayerNorm, but rolling our
# own teaches the two things that matter: nn.Parameter (a tensor the optimizer
# trains) and the biased variance (unbiased=False) that nn.LayerNorm uses.
# ----------------------------------------------------------------------------


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))   # learned scale
        self.beta = nn.Parameter(torch.zeros(d_model))   # learned shift
        self.eps = eps

    def forward(self, x):                                # x: (..., D)
        mean = x.mean(dim=-1, keepdim=True)              # (..., 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # biased — matches nn.LayerNorm
        return (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta


# ----------------------------------------------------------------------------
# Causal multi-head self-attention.
# ----------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide evenly into heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # one Linear per projection; bias-free, like the original transformer
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):                                # x: (B, T, D)
        B, T, D = x.shape
        H, hd = self.n_heads, self.d_head

        # project, split into heads, move the head axis next to the batch axis
        q = self.wq(x).view(B, T, H, hd).transpose(1, 2)   # (B, H, T, hd)
        k = self.wk(x).view(B, T, H, hd).transpose(1, 2)
        v = self.wv(x).view(B, T, H, hd).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)  # (B, H, T, T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))   # before softmax!
        attn = F.softmax(scores, dim=-1)                    # (B, H, T, T)

        out = attn @ v                                      # (B, H, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # merge heads
        return self.wo(out)                                 # (B, T, D)


# ----------------------------------------------------------------------------
# Position-wise MLP and the pre-LN transformer block.
# ----------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):                                # x: (B, T, D)
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):                                # pre-LN: normalize inside the branch
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ----------------------------------------------------------------------------
# The full model.
# ----------------------------------------------------------------------------


class GPT(nn.Module):
    def __init__(self, vocab, d_model=128, n_layers=3, n_heads=4, d_ff=512, max_T=256):
        super().__init__()
        self.max_T = max_T
        self.token_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_T, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.token_emb.weight         # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):                              # idx: (B, T) of token ids
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)         # (T,)
        x = self.token_emb(idx) + self.pos_emb(pos)      # (B, T, D), pos broadcasts over B
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)                              # (B, T, vocab)


@torch.no_grad()
def generate(model, idx, n_new, temperature=0.8):
    """Autoregressive sampling. idx: (B, T0) prompt -> (B, T0 + n_new)."""
    model.eval()
    for _ in range(n_new):
        idx_cond = idx[:, -model.max_T:]                 # never exceed the position table
        logits = model(idx_cond)[:, -1, :] / temperature  # (B, vocab): last position only
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)    # (B, 1)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------


def get_batch(data, batch_size, T, device, generator):
    """Sample batch_size random windows of length T+1 from the token array."""
    starts = torch.randint(0, data.shape[0] - T - 1, (batch_size,), generator=generator)
    idx = starts[:, None] + torch.arange(T + 1)[None, :]  # (B, T+1)
    return data[idx].to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenizer: every distinct character is a token
    chars = sorted(set(CORPUS))
    vocab = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in CORPUS], dtype=torch.long)
    print(f"corpus: {data.shape[0]} chars, vocab = {vocab}, device = {device}")

    model = GPT(vocab).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}  (expect first loss near ln {vocab} = "
          f"{math.log(vocab):.2f})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)       # reproducible batches

    model.train()
    t0 = time.time()
    recent = []
    for step in range(1, args.steps + 1):
        batch = get_batch(data, args.batch_size, args.seq_len, device, gen)
        inputs, targets = batch[:, :-1], batch[:, 1:]    # predict token t+1 from <= t

        logits = model(inputs)                           # (B, T, vocab)
        loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))

        opt.zero_grad()                                  # clear last step's grads
        loss.backward()                                  # fill .grad
        opt.step()                                       # update in place

        recent.append(loss.item())
        if step % 250 == 0 or step == 1:
            avg = sum(recent[-100:]) / len(recent[-100:])
            print(f"step {step:5d}  loss {avg:.3f}  ({time.time()-t0:.0f}s)")

    final = sum(recent[-100:]) / len(recent[-100:])
    print(f"\nFINAL: average loss over last 100 steps = {final:.3f}")

    prompt = torch.tensor([[stoi[c] for c in "Alice "]], dtype=torch.long, device=device)
    out = generate(model, prompt, 250)[0].tolist()
    print("\nsample:\n" + "".join(chars[i] for i in out))

    if final < 1.0:
        print(f"\nPASS: model fits the corpus (loss {final:.3f} < 1.0; "
              f"untrained baseline ln {vocab} = {math.log(vocab):.2f}).")
    else:
        print(f"\nFAIL: loss {final:.3f} did not reach 1.0.")


if __name__ == "__main__":
    main()
