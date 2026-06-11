"""PPO from scratch — reference implementation for the course (ch7.html).

Single-file PPO with a clipped surrogate objective, GAE, and a shared-trunk
actor-critic, in the style popularized by CleanRL. Section references (§) point
at the course chapters where each line is derived.

Run:  python ppo.py            # ~70 s on a laptop CPU; expect the PASS line
Deps: torch, gymnasium, numpy
"""

import argparse
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# --------------------------------------------------------------------------- #
# GAE (§3.5) — kept as a pure function so test_ppo_core.py can verify it.
# rewards, values, dones are arrays of shape (T, N); next_value (N,);
# next_done (N,). dones[t] / next_done mark *terminal* states (no bootstrap).
# Returns advantages (T, N); returns-to-fit = advantages + values (§7.2).
# --------------------------------------------------------------------------- #
def compute_gae(rewards, values, dones, next_value, next_done, gamma, lam):
    T = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros_like(next_value)
    for t in reversed(range(T)):
        if t == T - 1:
            not_terminal = 1.0 - next_done
            v_next = next_value
        else:
            not_terminal = 1.0 - dones[t + 1]
            v_next = values[t + 1]
        delta = rewards[t] + gamma * v_next * not_terminal - values[t]  # §3.3
        last_gae = delta + gamma * lam * not_terminal * last_gae        # §3.5
        advantages[t] = last_gae
    return advantages


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # Orthogonal init (§7.2): keeps early policy near-uniform, value near zero.
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),  # §7.2: tiny logits
        )

    def get_value(self, x):
        return self.critic(x).squeeze(-1)

    def get_action_and_value(self, x, action=None):
        dist = Categorical(logits=self.actor(x))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.get_value(x)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="CartPole-v1")
    p.add_argument("--total-timesteps", type=int, default=400_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--num-steps", type=int, default=128)     # rollout length T
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_timesteps // batch_size

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # SAME_STEP autoreset (§7.2): on episode end the env resets immediately and
    # hands back the new episode's first obs, with the dying state in
    # info["final_obs"]. With gymnasium's default NEXT_STEP mode the buffer
    # would silently absorb one garbage transition per episode.
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(args.env_id))
         for _ in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    n_actions = envs.single_action_space.n

    agent = Agent(obs_dim, n_actions)
    # eps=1e-5 (not the default 1e-8) is one of the standard PPO details (§7.2)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    obs_buf = np.zeros((args.num_steps, args.num_envs, obs_dim), dtype=np.float32)
    act_buf = np.zeros((args.num_steps, args.num_envs), dtype=np.int64)
    logp_buf = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    rew_buf = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    done_buf = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    val_buf = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32)
    next_done = np.zeros(args.num_envs, dtype=np.float32)

    global_step = 0
    start = time.time()
    recent_returns = []

    for update in range(1, num_updates + 1):
        # Learning-rate annealing (§7.2)
        frac = 1.0 - (update - 1.0) / num_updates
        optimizer.param_groups[0]["lr"] = frac * args.lr

        # ---------------- rollout (§7.1 phase 1): collect T steps ----------- #
        for t in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[t] = next_obs.numpy()
            done_buf[t] = next_done
            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(next_obs)
            act_buf[t] = action.numpy()
            logp_buf[t] = logp.numpy()
            val_buf[t] = value.numpy()

            next_obs_np, reward, terminated, truncated, info = envs.step(action.numpy())
            rew_buf[t] = reward
            # §7.2: 'terminated' = the MDP really ended (bootstrap with 0);
            # 'truncated' = a time limit cut the episode short — the state was
            # NOT terminal, so bootstrap with V(final_obs). Folding γ·V(s_T)
            # into the last reward implements that exactly, because GAE will
            # cut the recursion at this step anyway (done=1).
            for i in np.flatnonzero(truncated & ~terminated):
                final_obs = torch.tensor(info["final_obs"][i], dtype=torch.float32)
                with torch.no_grad():
                    rew_buf[t, i] += args.gamma * agent.get_value(final_obs).item()
            next_done = (terminated | truncated).astype(np.float32)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32)

            if "final_info" in info and "episode" in info["final_info"]:
                ep = info["final_info"]["episode"]
                for i, fin in enumerate(info["final_info"]["_episode"]):
                    if fin:
                        recent_returns.append(float(ep["r"][i]))

        # ---------------- GAE (§3.5) ---------------------------------------- #
        with torch.no_grad():
            next_value = agent.get_value(next_obs).numpy()
        advantages = compute_gae(rew_buf, val_buf, done_buf, next_value,
                                 next_done, args.gamma, args.gae_lambda)
        returns = advantages + val_buf  # TD(λ) value targets (§7.2)

        # Flatten (T, N, ...) -> (T*N, ...)
        b_obs = torch.tensor(obs_buf.reshape(-1, obs_dim))
        b_act = torch.tensor(act_buf.reshape(-1))
        b_logp = torch.tensor(logp_buf.reshape(-1))
        b_adv = torch.tensor(advantages.reshape(-1))
        b_ret = torch.tensor(returns.reshape(-1))
        b_val = torch.tensor(val_buf.reshape(-1))

        # ---------------- optimize (§7.1 phase 2): K epochs of minibatches -- #
        idx = np.arange(batch_size)
        clip_fracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(idx)
            for s in range(0, batch_size, minibatch_size):
                mb = idx[s:s + minibatch_size]

                _, new_logp, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb], b_act[mb])
                logratio = new_logp - b_logp[mb]
                ratio = logratio.exp()                       # r_t(θ), §6.2

                with torch.no_grad():
                    # k3 estimator of KL(old ‖ new) (§7.2 diagnostics)
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs.append(
                        ((ratio - 1.0).abs() > args.clip_eps).float().mean().item())

                mb_adv = b_adv[mb]
                # Advantage normalization per minibatch (§7.2)
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Clipped surrogate (§6.2) — written as a loss (negated)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - args.clip_eps, 1 + args.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Clipped value loss (§7.2)
                v_clipped = b_val[mb] + torch.clamp(
                    new_value - b_val[mb], -args.clip_eps, args.clip_eps)
                v_loss = 0.5 * torch.max(
                    (new_value - b_ret[mb]) ** 2,
                    (v_clipped - b_ret[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        if update % 10 == 0 or update == num_updates:
            avg = np.mean(recent_returns[-20:]) if recent_returns else float("nan")
            ev = 1 - np.var(b_ret.numpy() - b_val.numpy()) / (np.var(b_ret.numpy()) + 1e-8)
            print(f"update {update:4d}/{num_updates}  step {global_step:7d}  "
                  f"avg_return(last20) {avg:7.1f}  approx_kl {approx_kl:.4f}  "
                  f"clip_frac {np.mean(clip_fracs):.3f}  expl_var {ev:.2f}  "
                  f"sps {int(global_step / (time.time() - start))}")

    final = np.mean(recent_returns[-20:])
    print(f"FINAL: average return over last 20 episodes = {final:.1f}")
    if final >= 400:
        print("PASS: CartPole-v1 effectively solved (>= 400 / 500).")
    else:
        print("Below 400 — try more timesteps or another seed.")
    envs.close()


if __name__ == "__main__":
    main()
