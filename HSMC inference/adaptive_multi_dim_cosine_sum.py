# -*- coding: utf-8 -*-
"""
Performs inference on the frequencies of a multi-parameter binomial probability
distribution with success probability given by a sum of squared cosines: 
sum_i [ cos(theta_i*t/2)^2 ].

A sequential Monte Carlo approximation is used to represent the probability 
distributions, using Hamiltonian Monte Carlo and Metropolis-Hastings mutation 
steps.

Includes functions for both adaptive and non adaptive estimation. In the 
adaptive case, space occupation is used to make the measurement times increase
along with the increase in knowledge about the system.

The particle positions are plotted.
"""

import copy, itertools, random, importlib, matplotlib.pyplot as plt
from autograd import grad, numpy as np
import global_vars as glob
from modules.likelihoods import measure, likelihood, init_likelihoods
from modules.resampler import init_resampler, HMC_resampler, \
    print_resampler_stats
from modules.distributions import SMCparameters, plot_distribution, \
    generate_prior, sum_distributions, init_distributions, split_dict, \
    mean_cluster_variance, space_occupation
#np.seterr(all='warn')

reload = True
if reload:
    importlib.reload(sys.modules["global_vars"])
    importlib.reload(sys.modules["modules.likelihoods"])
    importlib.reload(sys.modules["modules.resampler"])
    importlib.reload(sys.modules["modules.distributions"])

np.seterr(all='warn')

glob.dim = 2
glob.lbound = np.zeros(glob.dim) # The left boundaries for the parameters.
glob.rbound = np.ones(glob.dim) # The right boundaries for the parameters.

N_particles = None # Number of samples used to represent the probability
#distribution, using a sequential Monte Carlo approximation.

measurements = None # Length of the data record (coincides with the number of 
#Bayes updates to be performed in the case where chunksize=1).

real_parameters = None

lbound = np.zeros(glob.dim) # The left boundaries for the parameters.
rbound = np.ones(glob.dim) # The right boundaries for the parameters.

first_bayes_update = True
def bayes_update(data, new, distribution, threshold, signal_resampling=False,
                 allow_repeated=True):
    '''
    Updates a prior distribution according to the outcome of a measurement, 
    using Bayes' rule. 
    
    Parameters
    ----------
    data: [([float],int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    new: slice
        The slice of the list `data` corresponding to freshly added data points
        (i.e. those which weren't contemplated in previous updates.
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The prior distribution (SMC approximation).
    threshold: float
        The threshold effective sample size that should trigger a resampling 
        step. 
    signal_resampling: bool, optional
        Whether to return a second variable denoting the ocurrence of 
        resampling (Default is False).
    allow_repeated: bool, optional
        Whether to allow repeated particles when resampling (Default is True).
        If False, extra Markov steps will be taken to ensure non-repetition when
        necessary, and alternative mutations will be used whem HMC is unlikely 
        to get an accepted proposal.
        
    Returns
    -------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The updated distribution (SMC approximation).
    ESS: float
        The effective sample size of `distribution` (inverse of the sum of 
        the particles' squared weights).
    resampled: bool
        Whether resampling has occurred.
    '''
    
    global first_bayes_update
    if first_bayes_update is True:
        rep = "; " if allow_repeated else "; not "
        rep += "allowing repeated particles"
        print("Bayes update: resampling threshold = ", threshold, rep)
        first_bayes_update = False

    N_particles = glob.N_particles
    acc_weight, acc_squared_weight = 0, 0
    
    # Perform a correction step by re-weighting the particles according to 
    #the last chunk of data added (i.e. to the ratio between the latest 
    #cumulative likelihood to the previous one, which cancels out all but those 
    #newest data).
    for key in distribution:
        particle = np.frombuffer(key,dtype='float64')
        new_weight = likelihood(data[new],particle)*distribution[key]
        distribution[key] = new_weight
        acc_weight += new_weight
    
    # Normalize the weights.
    for key in distribution:
        w = distribution[key]/acc_weight
        distribution[key] = w
        acc_squared_weight += w**2 # The inverse participation ratio will be
        #used to decide whether to resample.

    resampled = False
    ESS = 1/acc_squared_weight # Varies between 1 and N_particles.
    if (ESS <= threshold):
        resampled = True
        distribution,success = HMC_resampler(data, distribution, allow_repeated)
        if not success: 
            if signal_resampling:
                return distribution, ESS, None # Resampling interrupted midway.
            return distribution
        ESS = N_particles if not allow_repeated else \
                          1/sum(map(lambda x:x**2,list(distribution.values())))
    
    if signal_resampling:
        return distribution, ESS, resampled

    return distribution, ESS

first_offline_estimation = True
def offline_estimation(distribution, data, threshold=None, chunksize=1,
                       plot_all=False):
    '''
    Estimates a vector of parameters based on a given set of experiments 
    (times/controls and outcomes).
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The prior distribution (SMC approximation).
    data: [([float],int)]
        A vector of experimental results, each datum being of the form 
        (time,outcome), where 'time' is the control used for each experiment and 
        'outcome' is its result.
    threshold: float, optional
        The threshold effective sample size that should trigger a resampling 
        step when updating the distribution (Default is None, N_particles/2 
        will be used given the current value of the global variable 
        N_particles). 
    chunksize: int, optional
        The number of data to be added at each iteration; the cumulative data 
        makes up the target posterior to be sampled from at each step (Default 
        is 1).
    plot_all: bool, optional
        Whether to plot the particle positions at each step (Default is False).
        
    Returns
    -------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The final distribution (SMC approximation).
    '''
    global first_offline_estimation
    if first_offline_estimation is True:
        print("Estimation: data chunksize = ", chunksize)
        first_offline_estimation = False
    
    if len(data)==0:
        return
    
    if threshold is None:
        threshold = N_particles/2
        
    ans=""; resampled=False; counter=0; print("|0%",end="|")
    updates = len(data)//chunksize
    if updates < 10:
        progress_interval = 100/updates
    for i in range(updates):
        if plot_all:
            if updates>10:
                while ans!="Y" and ans!="N":
                    ans = input("\n> This is going to print over 10 graphs. "\
                                "Are you sure you want that?"\
                                " [offline_estimation]\n(Y/N)\n")
            else:
                ans = "Y"
            if ans=="Y":
                info = "- step %d" % (i)
                info += " [resampled]" if resampled else ""
                plot_distribution(distribution,real_parameters, 
                                  note=info)
            
        # Signal newest data for SMC weight updates.
        new = slice(i*chunksize,(i+1)*chunksize) 
        # Update the distribution: get the posterior of the current iteration, 
        #which is the prior for the next.
        distribution, ESS, resampled = bayes_update(data[0:((i+1)*chunksize)], 
                          new, distribution, threshold, signal_resampling=True) 
        if resampled is None:
            print("> Process interrupted at iteration %d due to non-invertible"
                  " covariance matrix. [offline_estimation]" % i,end="")
            if plot_all:
                plot_distribution(distribution,real_parameters, 
                  note="(interrupted; importance sampled but no Markov moves)")
            break
        
        # Print up to 10 progress updates spaced evenly through the loop.
        if updates < 10:
            counter+=progress_interval
            print(round(counter),"%",sep="",end="|")
        elif (i%(updates/10)<1): 
            counter+=10
            print(counter,"%",sep="",end="|")
    
        # Use the remaining data if applicable. This also ensures that all data 
        #will be used in a single update if the chunksize is greater than the 
        #total number of data. 
        if i==updates-1 and len(data)%chunksize!=0:
            new = slice(updates*chunksize,len(data))
            distribution,ESS = bayes_update(data, new, distribution, threshold) 
            
    print("") # For newline.
    return distribution

first_adaptive_estimation = True
def adaptive_estimation(distribution, updates, threshold=None, 
                        target_occupation = 0.0015, factor=1, exponent=0.95, 
                        plot_all=False):
    '''
    Estimates the vector of parameters by sequentially performing experiments
    whose controls (times) are chosen adaptively according to the previous 
    iteration's resulting distribution.
    The times are chosen to be inversely proportional to an estimate of fraction 
    of parameter space occupied by the particles, times the effective sample 
    size (normalized to 1). This is a measure of uncertainty, meant to increase
    the evolution times as more information has been gathered; the ESS is used
    to increase the evolution times in between resampling steps (since the 
    particle positions don't change during these periods, so neither does 
    occupation, and the chosen times would otherwise stay constant).
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The prior distribution (SMC approximation).
    updates: int
        The number of measurements/Bayes' updates to perform (these must match;
        each experiment's control is decided based on the latest posterior).
    threshold: float, optional
        The threshold effective sample size that should trigger a resampling 
        step when updating the distribution (Default is None, N_particles/2 
        will be used given the current value of the global variable 
        N_particles). 
    factor: float, optional
        The proportionality constant to be used for the adaptive times (Default 
        is 1).
    plot_all: bool, optional
        Whether to plot the particle positions at each step (Default is False).
        
    Returns
    -------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The final distribution (SMC approximation).
    data: [([float],int)]
        The vector of experimental results obtained and used for computing the
        target likelihoods the SMC algorithm sampled from. Each datum is of the 
        form (time,outcome), where 'time' is the (adaptive) control used for 
        each experiment and 'outcome' is its result.
    '''
    N_particles = glob.N_particles

    global first_adaptive_estimation
    if first_adaptive_estimation is True:
        print("Adaptive estimation: %d measurements/updates;"
        " times=%.1f/(occupation**%.2f*ESS/n); target_ocupation=%.5f" % 
        (updates,factor,exponent,target_occupation))
    
    if updates==0:
        return
    
    if threshold is None:
        threshold = N_particles/2
        
    ans=""; resampled=False; counter=0; print("|0%",end="|")
    data = []
    ESS = N_particles
    max_side = 50
    if updates < 10:
        progress_interval = 100/updates
    for i in range(updates):
        particles, _ = split_dict(distribution)
        occrate,max_side = space_occupation(particles,side=max_side)
        if occrate<=target_occupation:
            print("> Process interrupted at iteration %d due to having "
             "achieved target occupation. [offline_estimation]" % i,end="")
            break
        adaptive_time = factor*1/(occrate**exponent*ESS/N_particles)
        data.append((adaptive_time,measure(adaptive_time)))

        if plot_all:
            if updates>10:
                while ans!="Y" and ans!="N":
                    ans = input("> This is going to print over 10 graphs. "\
                                "Are you sure you want that?"\
                                " [offline_estimation]\n(Y/N)\n")
            else:
                ans = "Y"
            if ans=="Y":
                info = "- step %d" % (i)
                info += " [resampled]" if resampled else ""
                plot_distribution(distribution,real_parameters, 
                                  note=info)
        # The new data is just the last datum (chunksize=1 necessarily unlike 
        #offline).
        new = slice(-1,None)
        # Update the distribution: get the posterior of the current iteration, 
        #which is the prior for the next.
        distribution, ESS, resampled = bayes_update(data, 
                          new, distribution, threshold, signal_resampling=True) 
        if resampled is None:
            print("> Process interrupted at iteration %d due to non-invert"
                  "ible covariance matrix. [offline_estimation]" % i,end="")
            if plot_all:
                plot_distribution(distribution,real_parameters, 
                  note="(interrupted; importance sampled but no Markov moves)")
            break
        
        # Print up to 10 progress updates spaced evenly through the loop.
        if updates < 10:
            counter+=progress_interval
            print(round(counter),"%",sep="",end="|")
        elif (i%(updates/10)<1): 
            counter+=10
            print(counter,"%",sep="",end="|")
            
    print("") # For newline.

    if first_adaptive_estimation is True:
        print("Adaptive times: ", [t for t,r in data])
        first_adaptive_estimation = False
    return distribution, data

def main():
    global real_parameters, N_particles, measurements
    dim = glob.dim
    glob.measurements = 100

    # To initialize some constants (for printing information on the console):
    init_distributions() 
    init_likelihoods()
    init_resampler() # Also for the global statistics.
    
    random_parameters = False
    if random_parameters:
        glob.real_parameters = np.array([random.random() for d in range(dim)])
    else:
        glob.real_parameters = np.array([0.25,0.77]) 
        #real_parameters = np.array([0.25,0.77,0.40])
        #real_parameters = np.array([0.25,0.77,0.40,0.52])
    
    # For ease of use since we don't want to change these variables anymore:
    real_parameters, measurements = glob.real_parameters, glob.measurements
    
    test_resampling, test_no_resampling = True, False

    glob.N_particles = 20**dim #'N_particles' particles will be used for each 
    #group. Should be a power with integer base and exponent `dim` so the 
    #particles can be neatly arranged into a cubic latice for the prior (unless 
    #not using a uniform distribution/lattice).

    prior = generate_prior(distribution_type="uniform")
    groups = 1 # The algorithm will be ran independently for `groups`
    #particle groups, on the same data. Their results will be joined 
    #together in the end.
    
    if test_resampling:
        final_dists = []
        for i in range(groups):
            print("~ Particle group %d (of %d) ~" % (i+1,groups))
            dist, data = adaptive_estimation(copy.deepcopy(prior),measurements,
                          threshold=100,plot_all=False)
            final_dists.append(dist)
            
        # To get the correct statistics:
        glob.N_particles = glob.N_particles*groups 
        final_dist = sum_distributions(final_dists) if groups>1 \
                                                    else final_dists[0]
        plot_distribution(final_dist,real_parameters)
        mean_cluster_variance(final_dist,real_parameters)

    glob.print_info()
    print_resampler_stats()

    if test_no_resampling: 
        # For reference; use same data, but offline an no resampling.
        global first_bayes_update, first_offline_estimation
        first_bayes_update,first_offline_estimation = True, False
        # Chunksize is irrelevant here so don't print it (hence the False ^)
        glob.N_particles = 50**dim
        prior = generate_prior(distribution_type="uniform")
        dist_no_resampling = offline_estimation(copy.deepcopy(prior),data,
                                                threshold=0)
        plot_distribution(dist_no_resampling,real_parameters,
                          note="(no resampling)")
        print("No resampling test completed with %d particles." %
              glob.N_particles)
    
main()