# Quantum parameter estimation using Bayesian learning 
## (Alexandra's Master's project)

The folder **Phase estimation** contains implementations for estimating a phase using 2 different approaches: gaussian rejection filtering, and MCMC (with Hamiltonian Monte Carlo and Metropolis-Hastings transitions).

The folder **Precession frequency estimation** contains implementations for estimating a precession frequency (plus a decay factor in some cases) using 3 different approaches: SMC (with sequential importance resampling), HSMC (relying mostly on Hamiltonian Monte Carlo mutation steps) and MCMC (using Hamiltonian Monte Carlo and Metropolis-Hastings transitions).

The folder **Sampling methods** contains implementations of several Monte Carlo sampling algorithms for a 3 target probability densities (a Rosenbrock function, a 6-dimensional gaussian, and a smiley face kernel density estimate).
