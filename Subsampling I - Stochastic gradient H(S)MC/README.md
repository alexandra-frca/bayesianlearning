Some exercises based on Stochastic Gradient Hamiltonian Monte Carlo[https://arxiv.org/pdf/1402.4102.pdf].

The file `SG-HMC_oscillator` simulates Hamiltonian dynamics with injected noise, both correcting and not for it by introducing friction (inspired on the Langevin equation).

The file `SG-HSMC_multi_dim_cosine_sum` applies subsampling to an inference problem (targeting the same univariate multi-parameter probabilistic function as `HSMC inference/multi_dim_cosine_sum.py`), again compensating for the (here subsampling induced) noise with friction. 
