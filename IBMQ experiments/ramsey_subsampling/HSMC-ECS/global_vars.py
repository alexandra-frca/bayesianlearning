# -*- coding: utf-8 -*-
"""
Module holding the variables that must be accessible from two or more modules.
"""
dim = None # Dimension of the parameter space.

N_particles = None # Number of samples used to represent the probability
#distribution, using a sequential Monte Carlo approximation.

measurements = None # Length of the data record.

samples = None # Number of samples to consider for each particle and iteration
#when subsampling.

real_parameters = None # The vector of real parameters (inference subject).

lbound = None # The left boundaries for the parameters.
rbound = None # The right boundaries for the parameters.