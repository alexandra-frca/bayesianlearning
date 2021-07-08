# Quantum parameter estimation using Bayesian learning 
## (Alexandra's Master's project)

The **HSMC inference** folder contains the application of Hamiltonian sequential Monte Carlo to the characterization of multi-parameter probabilistic functions (using sequential importance resampling, with the prior distribution as importance function).

The **HSMC-ECS (annealing)** folder implements energy conserving subsampling HSMC, using a block pseudo marginal approach (in a tempered likelihood setting).

The **IBMQ experiments** folder contains a Jupyter notebook with several characterization experiments on IBMQ's backends (performed using *OpenPulse*), along with a selection of the data gathered from them.

The **Phase estimation** folder contains scripts that estimate a phase using 2 different approaches: gaussian rejection filtering, and MCMC (with Hamiltonian Monte Carlo and random walk Metropolis transitions).

The **Precession frequency estimation** folder contains implementations of the estimation of a precession frequency (plus a decay factor in some cases) using 3 different approaches: SMC (with sequential importance resampling), HSMC (relying mostly on Hamiltonian Monte Carlo mutation steps) and MCMC (using Hamiltonian Monte Carlo and random walk Metropolis transitions).

The **Sampling methods** folder contains implementations of several Monte Carlo sampling algorithms for 3 target probability densities (a Rosenbrock function, a 6-dimensional gaussian, and a smiley face kernel density estimate).

The **SG-H(S)MC (annealing)** folder implements Hamiltonian dynamics based algorithms accounting for noisy gradients - e.g. from subsampling, namely stochastic gradient HMC-based SMC (in a tempered likelihood setting).

The **Tempered likelihood HSMC inference** folder contains the application of tempered/annealed Hamiltonian sequential Monte Carlo to the characterization of multi-parameter probabilistic functions.

The **Trajectory sampling strategies** folder applies several variations of HMC to a Rosenbrock function density. These variations differ in how they pick a sample from the generated trajectory (in all other files only the last state was considered, and a Metropolis correction was performed). Includes static and dynamic approaches.

The `dissertation_examples` jupyter notebook visually represents a few key examples to be used for illustrations.
