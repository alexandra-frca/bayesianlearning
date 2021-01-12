## Precession frequency estimation

The most recently modified files are `precession_2d.py` and `precession_2d_SHMC.py`, both of which estimate a precession frequency plus a decay factor using respectively a sequential Monte Carlo approximation (with sequential importance resampling) and a sequential Hamiltonian Monte Carlo approximation.

`precession_1d.py` and `precession_1d_SHMC.py` estimate the frequency only, using the same methods as before but assuming a decoherence-free system.

These 4 scripts repeat the algorithm for a number of runs with randomly picked real values, and medians are taken over all of them to get the results. The evolution of the median standard deviations with the iterations is plotted.

The folder **Single Runs** contains some scripts for generating graphs respecting single runs of the 1d version; namely, final cumulative distribution functions for bimodal distributions (i.e. the prior is changed to be an even function, with support over negative frequencies) and error bar plots for some real frequency of choice.
