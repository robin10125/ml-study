# ML-Study
 
A collection of study apps for learning machine learning concepts.

- `flash-attention-course/` — 7-chapter active-recall course on Flash Attention (CUDA + JAX companion code).
- `ppo-course/` — 7-chapter active-recall course on PPO, from REINFORCE through TRPO to a from-scratch implementation (`code/ppo.py` solves CartPole; `code/test_ppo_core.py` machine-checks the math).
- `jax-transformer-course/` — 7-chapter active-recall course teaching JAX from zero (for PyTorch/TF users) through defining and training a transformer LM without reference (`code/transformer.py` trains and PASSes; `code/test_jax_core.py` machine-checks the math).
- `kl-divergence.html` — single-page lesson.
- `transformer-jax.html` — single-page advanced sequel to the JAX course: RoPE, grouped-query attention, Mixture-of-Experts.

Open any course's `index.html` in a browser; progress is stored in localStorage.

Made with Claude.
