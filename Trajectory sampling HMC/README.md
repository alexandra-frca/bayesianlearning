All scripts consider the same target density: a Rosenbrock *smiley* function (same as `Sampling methods/Rosenbrock Function`). 

The file `rosenbrock_last_state.py` implements random walk Metropolis and/or the typical version of HMC, where the sample is always chosen to be the position corresponding to the last state of the integrated trajectory and a Metropolis-Hastings acceptance/rejection step is performed for correctness.

The file `rosenbrock_no_u_turn_sampler.py` is a dynamic implementation of HMC, where the number of integration time-steps parameter L is picked adaptively based on a termination criterion. This is based on  [The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo](https://arxiv.org/pdf/1111.4246.pdf) (but with multinomial sampling instead of slice sampling and no energy restrictions) and [A Conceptual Introduction to Hamiltonian Monte Carlo](https://arxiv.org/pdf/1701.02434.pdf).

The file `rosenbrock_progressive_sampling.py` uses either the last state or static progressive sampling; the latter can be chosen to be uniform or partly or fully biased (these 2 favour more distant proposals while preserving correctness).
