Based on [Hamiltonian Monte Carlo with Energy Conserving Subsampling](https://jmlr.csail.mit.edu/papers/volume20/17-452/17-452.pdf), [Subsampling Sequential Monte Carlo for Static Bayesian Models](https://arxiv.org/pdf/1805.03317.pdf) and [The Block Pseudo-Marginal Sampler](https://arxiv.org/pdf/1603.02485.pdf).

The file `HSMC-ECS_multi_dim_cosine_sum.py` applies subsampling to an inference problem (targeting the same univariate multi-parameter probabilistic function as `HSMC inference/multi_dim_cosine_sum.py`), using a block pseudo-marginal approach to get fully correct Hamiltonian dynamics.

The module `global_vars.py` holds the variables that need to be accessible from two or more modules, the package `function_evaluations` contains all functions related to estimators and likelihood functions, and the package `tools` contains all resampling, distribution and statistics related functions. 
