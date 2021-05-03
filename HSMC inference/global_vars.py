# -*- coding: utf-8 -*-
"""
Module holding the variables that must be accessible from two or more modules.
"""
dim = None # Dimension of the parameter space.

N_particles = None # Number of samples used to represent the probability
#distribution, using a sequential Monte Carlo approximation.

measurements = None # Length of the data record.

real_parameters = None # The vector of real parameters (inference subject).

lbound = None # The left boundaries for the parameters.
rbound = None # The right boundaries for the parameters.

def print_info():
    '''
    Prints some relevant information relative to the run of the algorithm on
    the console.
    ''' 
    print("> n=%.2f^%d; N=%d; %dd sum of squared cosines " 
          % (N_particles**(1/dim),dim,measurements,dim))