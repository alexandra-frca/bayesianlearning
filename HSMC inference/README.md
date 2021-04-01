The file `multi_dim_cosine_sum.py` performs inference on the *n* parameters of a univariate probability distribution given by a normalized sum of *n* squared cosines. 

The file `multivariate_cosine.py` does the same, but for a multivariate distribution consisting of a binomial distribution along each dimension (with the probability of success a squared cosine).

In both cases, for *dim=1* the problem reduces to that of `Precession frequency estimation/precession_1d_HSMC.py`, apart from some differences in the implementation and plotted quantities.

The file `ocupation_rate.py` calculates the parameter space occupation rate - as an inverse measure of dispersion -, and plots the particles and grid used for the computation (for some test scenario in the 2-d case).
