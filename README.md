# ML-Study
 
A collection of study apps for learning machine learning concepts.

- `flash-attention-course/` — 7-chapter active-recall course on Flash Attention (CUDA + JAX companion code).
- `ppo-course/` — 7-chapter textbook + problem-set course on PPO, from REINFORCE through TRPO to a from-scratch implementation; problems (derivations and code) are written in in-page workspaces with gated reference solutions (`code/ppo.py` solves CartPole; `code/test_ppo_core.py` machine-checks the math).
- `jax-transformer-course/` — 7-chapter textbook + problem-set course teaching JAX from zero (for PyTorch/TF users) through defining and training a transformer LM without reference; problems are written in in-page code workspaces with gated reference solutions (`code/transformer.py` trains and PASSes; `code/test_jax_core.py` machine-checks the math).
- `pytorch-transformer-course/` — 7-chapter textbook + problem-set course teaching PyTorch from zero (the *syntax*, not the architecture) through writing a GPT-style transformer from an empty file: tensors/shapes, autograd, `nn.Module`, the training loop, then the model and its training/sampling; 35 in-page code problems with gated reference solutions (`code/transformer.py` memorizes a corpus to loss 0.093 and PASSes; `code/test_torch_core.py` machine-checks the components — `ALL 7 CHECKS PASS`).
- `kl-divergence.html` — single-page lesson.
- `transformer-jax.html` — single-page advanced sequel to the JAX course: RoPE, grouped-query attention, Mixture-of-Experts.

Open any course's `index.html` in a browser; progress is stored in localStorage.

Made with Claude.
