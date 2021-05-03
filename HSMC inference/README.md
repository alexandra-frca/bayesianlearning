The file `multi_dim_cosine_sum.py` performs inference on the *n* parameters of a univariate probability distribution given by a normalized sum of *n* squared cosines. 

The file `multivariate_cosine.py` does the same, but for a multivariate distribution consisting of a binomial distribution along each dimension (with the probability of success a squared cosine).

In both cases, for *dim=1* the problem reduces to that of `Precession frequency estimation/precession_1d_HSMC.py`, apart from some differences in the implementation and plotted quantities.

The folder **2d cosine 3d plot** contains data from a run of `multi_dim_cosine_sum.py` (a vector of controls and outcomes + a final SMC distribution) and a notebook for plotting the particles of the distribution juxtaposed against the probability density function (i.e. the likelihood arising from the vector of data the SMC algorithm sampled from). A sample graph is included in the slides. 

The file `ocupation_rate.py` calculates the parameter space occupation rate - as an inverse measure of dispersion -, and plots the particles and grid used for the computation (for some test scenario in the 2-d case).

The file `adaptive_multi_dim_cosine_sum.py` uses the space occupation construction to choose the measurement times adaptively (as opposed to picking them randomly and offline as in `multi_dim_cosine_sum.py`).

The file `incremental_multi_dim_cosine_sum.py` performs offline inference while tendentially increasing the evolution times, adapting the inference process of `multi_dim_cosine_sum.py` to loosely reproduce the trend of the adaptive strategy. 

The file `variance_calculations.py` provides a quantitative estimate of the uncertainty of the distributions resulting from the schemes above by averaging variances.

The `modules` folder and the `global_vars.py` module gather respectively function modules and variables that are common to `multi_dim_cosine_sum.py`, `adaptive_multi_dim_cosine_sum.py` and `incremental_multi_dim_cosine_sum.py`.
