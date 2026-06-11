"""Machine checks for the PPO building blocks taught in the course.

Each test verifies a fact the chapters ask you to derive by hand:
  1. GAE recursion == brute-force double sum          (§3.5)
  2. λ=1  =>  advantage = discounted return − V(s)    (§3.5)
  3. λ=0  =>  advantage = one-step TD error           (§3.5)
  4. Clipped surrogate on a hand-computed example     (§6.2)
  5. Clipping kills the gradient exactly when it should (§6.3)
  6. Baseline does not bias the policy gradient: E[∇log π] = 0 (§3.1)

Run:  python test_ppo_core.py     (expect 6 PASS lines)
"""

import numpy as np
import torch

from ppo import compute_gae

rng = np.random.default_rng(0)


def brute_force_gae(rewards, values, dones, next_value, next_done, gamma, lam):
    """GAE straight from the definition: A_t = sum_l (γλ)^l δ_{t+l},
    with the sum cut at terminal states."""
    T, N = rewards.shape
    v_next = np.concatenate([values[1:], next_value[None]], axis=0)
    d_next = np.concatenate([dones[1:], next_done[None]], axis=0)
    deltas = rewards + gamma * v_next * (1 - d_next) - values
    adv = np.zeros_like(rewards)
    for n in range(N):
        for t in range(T):
            acc, coef = 0.0, 1.0
            for l in range(t, T):
                acc += coef * deltas[l, n]
                if (l < T - 1 and dones[l + 1, n]) or (l == T - 1 and next_done[n]):
                    break  # episode boundary: later deltas belong to a new episode
                coef *= gamma * lam
            adv[t, n] = acc
    return adv


def test_gae_matches_definition():
    T, N = 40, 3
    rewards = rng.normal(size=(T, N)).astype(np.float64)
    values = rng.normal(size=(T, N))
    dones = (rng.random((T, N)) < 0.1).astype(np.float64)
    dones[0] = 0
    next_value = rng.normal(size=N)
    next_done = np.array([0.0, 1.0, 0.0])
    a1 = compute_gae(rewards, values, dones, next_value, next_done, 0.99, 0.95)
    a2 = brute_force_gae(rewards, values, dones, next_value, next_done, 0.99, 0.95)
    assert np.allclose(a1, a2, atol=1e-10), np.abs(a1 - a2).max()
    print("PASS 1: GAE backward recursion == brute-force sum of (γλ)^l δ")


def test_lambda_one_is_mc_minus_v():
    T, N, gamma = 30, 2, 0.97
    rewards = rng.normal(size=(T, N))
    values = rng.normal(size=(T, N))
    dones = np.zeros((T, N))
    next_value = rng.normal(size=N)
    next_done = np.zeros(N)
    adv = compute_gae(rewards, values, dones, next_value, next_done, gamma, 1.0)
    # Discounted return bootstrapped with next_value at the horizon
    ret = np.zeros((T, N))
    run = next_value.copy()
    for t in reversed(range(T)):
        run = rewards[t] + gamma * run
        ret[t] = run
    assert np.allclose(adv, ret - values, atol=1e-10)
    print("PASS 2: λ=1 gives A_t = (discounted return) − V(s_t)  (pure Monte Carlo)")


def test_lambda_zero_is_td_error():
    T, N, gamma = 25, 2, 0.99
    rewards = rng.normal(size=(T, N))
    values = rng.normal(size=(T, N))
    dones = np.zeros((T, N))
    next_value = rng.normal(size=N)
    next_done = np.zeros(N)
    adv = compute_gae(rewards, values, dones, next_value, next_done, gamma, 0.0)
    v_next = np.concatenate([values[1:], next_value[None]], axis=0)
    delta = rewards + gamma * v_next - values
    assert np.allclose(adv, delta, atol=1e-10)
    print("PASS 3: λ=0 gives A_t = δ_t = r_t + γV(s_{t+1}) − V(s_t)  (one-step TD)")


def test_clipped_loss_hand_example():
    # Hand-checkable example from ch6: ε = 0.2
    ratio = torch.tensor([1.5, 0.5, 1.1, 0.7])
    adv = torch.tensor([1.0, 1.0, -2.0, -1.0])
    # per-sample objective: min(r·A, clip(r)·A)
    #  r=1.5, A=+1: min(1.5, 1.2·1)   = 1.2   (clipped above)
    #  r=0.5, A=+1: min(0.5, 0.8·1)   = 0.5   (unclipped: min picks the worse)
    #  r=1.1, A=−2: min(−2.2, −2.2)   = −2.2  (clip inactive inside [0.8,1.2])
    #  r=0.7, A=−1: min(−0.7, −0.8)   = −0.8  (clip pessimizes to 0.8)
    obj = torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv)
    expected = torch.tensor([1.2, 0.5, -2.2, -0.8])
    assert torch.allclose(obj, expected)
    print("PASS 4: clipped surrogate matches the hand-computed ch6 example")


def test_clip_gradient_regions():
    eps = 0.2
    # Case A: positive advantage, ratio above 1+ε  -> gradient must be zero
    logits_shift = torch.tensor([0.5], requires_grad=True)
    ratio = torch.exp(logits_shift)            # e^0.5 ≈ 1.65 > 1.2
    obj = torch.min(ratio * 1.0, torch.clamp(ratio, 1 - eps, 1 + eps) * 1.0)
    obj.backward()
    assert logits_shift.grad.abs().item() == 0.0
    # Case B: negative advantage, ratio above 1+ε -> min picks the UNCLIPPED
    # branch (more negative), so the gradient is alive and pushes ratio down.
    logits_shift = torch.tensor([0.5], requires_grad=True)
    ratio = torch.exp(logits_shift)
    obj = torch.min(ratio * -1.0, torch.clamp(ratio, 1 - eps, 1 + eps) * -1.0)
    obj.backward()
    assert logits_shift.grad.abs().item() > 0.0
    print("PASS 5: grad is zero when clip protects (A>0, r>1+ε); alive when "
          "the unclipped branch is worse (A<0, r>1+ε)")


def test_eglp_score_function_mean_zero():
    # E_{a~π}[∇θ log πθ(a)] = 0 — the lemma that makes baselines unbiased.
    torch.manual_seed(0)
    logits = torch.randn(5, requires_grad=True)
    probs = torch.softmax(logits, dim=0)
    # exact expectation over the 5 actions instead of sampling
    total = torch.zeros(5)
    for a in range(5):
        logits.grad = None
        logp = torch.log_softmax(logits, dim=0)[a]
        logp.backward(retain_graph=True)
        total = total + probs[a].detach() * logits.grad
    assert total.abs().max().item() < 1e-7
    print("PASS 6: E[∇ log π] = 0 (EGLP lemma) verified exactly for a categorical")


if __name__ == "__main__":
    test_gae_matches_definition()
    test_lambda_one_is_mc_minus_v()
    test_lambda_zero_is_td_error()
    test_clipped_loss_hand_example()
    test_clip_gradient_regions()
    test_eglp_score_function_mean_zero()
    print("ALL 6 CHECKS PASS")
