# Quantum parameter estimation using Bayesian learning 
## (Alexandra's Master's project)

The most recently modified files are `precession_2d.py` and `precession_2d_SHMC.py`, both of which estimate a precession frequency plus a decay factor using respectively a sequential Monte Carlo approximation (with sequential importance resampling) and a sequential Hamiltonian Monte Carlo approximation.

`precession_1d.py` and `precession_1d_SHMC.py` estimate the frequency only, using the same methods as before but assuming a decoherence-free system.

`likelihoodfree.py` is similar to `precession_1d.py `, but doesn't rely on the reconstruction of the likelihood function 
(requiring only a sample outcome per particle per step).

The folder **Simulation Methods** contains implementations of several Monte Carlo sampling algorithms for a set of target probability densities.
