The file `tempered_likelihood_HSMC.py` performs inference on the *n* parameters of a univariate probability distribution given by a normalized sum of *n* squared cosines, using tempered/annealed HSMC.

The problem is the same as in `HSMC inference/multi_dim_cosine_sum.py`, where instead of annealed likelihoods the sequence of target distributions was obtained by sequentially adding chunks of data to the (cumulative) data record.

As before, for *dim=1* the problem reduces to that of `Precession frequency estimation/precession_1d_HSMC.py`, apart from some differences in the implementation and plotted quantities.
