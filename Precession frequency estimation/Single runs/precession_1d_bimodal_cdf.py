# -*- coding: utf-8 -*-
"""
Hamiltonian learning implementation for a simple precession example, using both
offline and adaptive Bayesian inference.

The qubit is assumed to be initialized at state |+> for each iteration, and to
evolve under H = f*sigma_z/2, where f is the parameter to be estimated.

A sequential Monte Carlo approximation is used to represent the probability 
distributions, along with Liu-West resampling.
"""

import sys, random, numpy as np, matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

N_particles = 2000 # Number of samples used to represent the probability
#distribution, using a sequential Monte Carlo approximation.

f_real = 0 # The actual precession frequency we mean to estimate.

alpha_real = 0 # The decay factor (the inverse of the coherence time).

def measure(t, alpha=alpha_real, tries=1):
    '''
    Simulates the measurement of the quantum system of the x component of spin 
    at a given time t after initialization at state |+>.
    
    Parameters
    ----------
    t: float
        The evolution time between the initialization and the projection.
    alpha: int, optional
        The exponential decay parameter (Default is 0).
    tries: int, optional
        The amount of times the measurement is repeated (Default is 1).
        
    Returns
    -------
    1 if the result is |+>, 0 if it is |->.
    '''
    global f_real
    r = random.random()
    p = np.random.binomial(tries, 
                           p=(np.cos(f_real*t/2)**2*np.exp(-alpha*t)+
                              (1-np.exp(-alpha*t))/2))/tries
    if (r<p):
        return 1
    return 0

def simulate(test_f, t):
    '''
    Provides an estimate for the likelihood  P(D=1|test_f,t) of an x-spin 
    measurement at time t yielding result |+>, given a test parameter for the 
    fixed form Hamiltonian. 
    This estimate is computed as the fraction of times a simulated system
    evolution up to time t yields said result upon measurement.
    
    Parameters
    ----------
    test_f: float
        The test precession frequency.
    t: float
        The evolution time between the initialization and the projection.
        
    Returns
    -------
    p: float
        The estimated probability of finding the particle at state |+>.
    '''
    p=np.cos(test_f*t/2)**2
    return p 

def resample(f_particle, distribution, resampled_distribution, 
             a=0.98, left_constraint = -10, right_constraint=10):
    '''
    Resamples a particle from a given amount of times and adds the results to 
    a previous distribution. Uniform weights are attributed to the resampled 
    particles; the distribution is caracterized by particle density.
    For a=0, the resampling works as a bootstrap filter.
    
    Parameters
    ----------
    f_particle: float
        The frequency particle to be resampled.
    distribution: dict
        , with (key,value):=(frequency particle,importance weight)
        The prior distribution (SMC approximation).
    resampled_distribution: dict
        , with (key,value):=(frequency particle,importance weight)
        The distribution (SMC approximation) in the process of being resampled. 
        When returning, it will have been updated with the freshly resampled 
        particle(s).
    a: float, optional
        The Liu-West filtering parameter (Default is 0.98).
    '''
    current_SMCmean, current_stdev = SMCparameters(distribution)
    # Start with any invalid particle.
    new_particle = left_constraint-1
    
    # Sample a valid frequency particle.
    while (new_particle < left_constraint or new_particle > right_constraint):
        mean = a*f_particle+(1-a)*current_SMCmean
        stdev = (1-a**2)**0.5*current_stdev
        new_particle = np.random.normal(mean,scale=stdev)
        # Avoid repeated particles for variety.
        if new_particle in resampled_distribution:
            new_particle = left_constraint-1
    # Attribute uniform weights to the resampled particles.
    resampled_distribution[new_particle] = 1/N_particles
    #smooth_and_plot(cumulative_distribution_function(resampled_distribution))

def bayes_update(distribution, t, outcome, threshold=N_particles/2):
    '''
    Updates a prior distribution according to the outcome of a measurement, 
    using Bayes' rule. 
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(frequency particle,importance weight)
        The prior distribution (SMC approximation). When returning, it will 
        have been updated according to the provided experimental datum.
        
    t: float
        The time at which the measurement was performed, which characterizes 
        it. The reference (t=0) is taken as the time of initialization.
    outcome: int
        The result of the measurement (with |+> mapped to 1, and |-> to 0).
    threshold: float, optional
        The threshold inverse participation ratio for resampling 
        (Default is N_particles/2). For threshold=0, no resampling takes place.
    '''
    global N_particles
    acc_weight = 0
    acc_squared_weight = 0
    
    # Calculate the weights.
    for particle in distribution:
        p1 = simulate(particle,t)
        if (outcome==1):
            w = p1*distribution[particle]
        if (outcome==0):
            w = (1-p1)*distribution[particle]
        distribution[particle] = w
        acc_weight += w
    
    # Normalize the weights.
    for particle in distribution:
        w = distribution[particle]/acc_weight
        distribution[particle] = w
        acc_squared_weight += w**2 # The inverse participation ratio will be
        #used to decide whether to resample or not.
       
    # Check whether the resampling condition applies, and resample if so.
    if (1/acc_squared_weight <= threshold):
        #print("dist",distribution)
        #print("mean,stdev",SMCparameters(distribution))
        resampled_distribution = {}
        #print(SMCparameters(distribution))
        for i in range(N_particles):
            particle = random.choices(list(distribution.keys()), 
                                  weights=distribution.values())[0]
            resample(particle,distribution,
                     resampled_distribution)
        distribution = resampled_distribution
        #print("res", distribution)
        #smooth_and_plot(cumulative_distribution_function(distribution))
        
    return(distribution)

def SMCparameters(distribution, stdev=True):
    '''
    Calculates the mean and (optionally) standard deviation of a given 
    distribution.
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(frequency particle,importance weight)
        The distribution (SMC approximation) whose parameters are to be 
        calculated.
    stdev: bool
        To be set to False if the standard deviation is not to be returned 
        (Default is True).
        
    Returns
    -------
    mean: float
        The mean of the distribution.
    stdev: float
        The standard deviation of the distribution.
    '''
    mean = 0
    meansquare = 0
    for particle in distribution:
        f = particle
        w = distribution[particle]
        mean += f*w
        meansquare += f**2*w
    if not stdev:
        return mean
    stdev = abs(mean**2-meansquare)**0.5
    return mean,stdev

def offline_estimation(distribution, f_max, steps, increment=0.08):
    '''
    Estimates the precession frequency by defining a set of experiments, 
    performing them, and updating a given prior distribution according to their 
    outcomes (using Bayesian inference).
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(frequency particle,importance weight)
        The prior distribution (SMC approximation).
    f_max: float
        The upper limit for the expected frequency (used to abide by Nyquist's 
        theorem).
    steps: int
        The total number of experiments to be performed.
        
    Returns
    -------
    mean: float
        The mean of the final distribution.
    stdev: float
        The standard deviation of the final distribution.
    means: [float]
        A list of the consecutive distribution means, including the prior's and
        the ones resulting from every intermediate step.
    stdevs: [float]
        A list of the consecutive distribution standard deviations, including 
        the prior's and the ones resulting from every intermediate step.
    increment: float, optional
        The increment between consecutive measurement times (Default is 0.08).
    '''
    # We'll use evenly spaced times, fixed in advance.
    mean, stdev = SMCparameters(distribution)
    means, stdevs = [], []
    means.append(mean)
    stdevs.append(stdev) 
    
    ts = np.arange(1, steps+1)*increment
    
    for t in ts:
        m = measure(t)
        # Update the distribution: get the posterior of the current iteration, 
        #which is the prior for the next.
        distribution = bayes_update(distribution,t,m) 
        #smooth_and_plot(cumulative_distribution_function(distribution))
        
        mean,stdev = SMCparameters(distribution)
        means.append(mean)
        stdevs.append(stdev) 
        
    cumulative_times = np.cumsum([i/(2*f_max) for i in range(steps+1)])
    return means, stdevs, cumulative_times, distribution
    
def expected_utility(distribution, time):
    '''
    Returns the expectation value for the utility of a measurement time.
    The utility function considered is a weighed sum of the negative variances
    of the two parameters, adjusted for their relative scale.
    Its expectation value is computed over all outcomes, given the current
    distribution.
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        The prior distribution (SMC approximation).
    time: float
        The measurement time for which the utility is to be computed.
        
    Returns
    -------
    utility: float
        The expectation value for the utility of the provided measurement time. 
    '''
    # Obtain the probability of each oucome, given the current distribution.
    p1=0
    for particle in distribution:
        p1 += simulate(particle,time)*distribution[particle]
    p0 = 1-p1
        
    dist_0 = distribution.copy()
    dist_1 = distribution.copy()
    
    # Update the distribution assuming each of the possible outcomes.
    dist_0 = bayes_update(dist_0,time,0) 
    dist_1 = bayes_update(dist_1,time,1)
    
    # Compute the expected utility for each oucome.
    stdevs_0 = SMCparameters(dist_0)[1]
    stdevs_1 = SMCparameters(dist_1)[1]
    util_0 = -stdevs_0**2
    util_1 = -stdevs_1**2
    
    # Calculate the expected utility over all (both) outcomes.
    utility = p0*util_0 + p1*util_1
    
    return(utility)

def adaptive_guess(distribution, k, guesses):
    '''
    Provides a guess for the evolution time to be used for a measurement,
    picked using the PGH, a particle guess heuristic (where the times are 
    chosen to be inverse to the distance between two particles sampled at 
    random from the distribution).
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        The prior distribution (SMC approximation).
    k: float
        The proportionality constant to be used for the particle guess 
        heuristic.
    guesses: int
        The amount of hypothesis to be picked for the time using the PGH; only 
        the one which maximizes the expected utility among this set will be  
        chosen (Default is 1).
        
    Returns
    -------
    adaptive_t: float
        The measurement time to be used.
    '''
    adaptive_ts, utilities = [], []
    for i in range(guesses):
        delta=0
        while (delta==0):
            [f1, f2] = random.choices(list(distribution.keys()), 
                                      weights=distribution.values(), k=2)
            delta = abs(f1-f2)
        time = k/delta
        if (guesses==1):
            return(time)
        adaptive_ts.append(time)
    for t in adaptive_ts:
        utilities.append(expected_utility(distribution,t))
    return(adaptive_ts[np.argmax(utilities)])

def adaptive_estimation(distribution, steps, precision=0, k=1.25, guesses=1):
    '''
    Estimates the precession frequency by adaptively performing a set of 
    experiments, using the outcome of each to update the prior distribution 
    (using Bayesian inference) as well as to decide the next experiment to be 
    performed.
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(frequency particle,importance weight)
        The prior distribution (SMC approximation).
    steps: int, optional
        The maximum number of experiments to be performed.
    precision: float
        The threshold precision required to stop the learning process before  
        attaining the step number limit (Default is 0).
    k: float
        The proportionality constant to be used for the particle guess 
        heuristic (Default is 1.25).
        
    Returns
    -------
    mean: float
        The mean of the final distribution.
    stdev: float
        The standard deviation of the final distribution.
    means: [float]
        A list of the consecutive distribution means, including the prior's and
        the ones resulting from every intermediate step.
    stdevs: [float]
        A list of the consecutive distribution standard deviations, including 
        the prior's and the ones resulting from every intermediate step.
    '''
    mean, stdev = SMCparameters(distribution)
    means, stdevs, cumulative_times = [], [], []
    means.append(mean)
    stdevs.append(stdev)
    cumulative_times.append(0)
    
    if (guesses==1):
        adaptive_t = k/stdev
    else:
        adaptive_t = adaptive_guess(distribution,k,guesses)
        
    cumulative_times.append(adaptive_t)
        
    for i in range(1,steps+1):
        m = measure(adaptive_t)
        # Update the distribution: get the posterior of the current iteration, 
        #which is the prior for the next.
        distribution = bayes_update(distribution,adaptive_t,m)
        
        mean,stdev = SMCparameters(distribution)
        means.append(mean)
        stdevs.append(stdev) 
        
        if (stdev <= precision): 
            for j in range(i+1,steps+1): # Fill in the rest to get fixed length
        #lists for the plots.
                means.append(mean)
                stdevs.append(stdev) 
                cumulative_times.append(cumulative_times[i-1])
            break
            
        if (guesses==1):
            adaptive_t = k/stdev
        else:
            adaptive_t = adaptive_guess(distribution,k, guesses)
            
        cumulative_times.append(adaptive_t+cumulative_times[i-1])
    return means, stdevs, cumulative_times, distribution

def cumulative_distribution_function(distribution):
    sorted_particles = sorted([key for key in list(distribution.keys())])
    cumulative_sum = 0
    points=[]
    for particle in sorted_particles:
        cumulative_sum += distribution[particle]
        points.append((particle,cumulative_sum))
    return points

def smooth_and_plot(points,smoothen=False, vertical_lines=[],x_steps=100):
    x,y = [[point[i] for point in points] for i in range(2)]
    
    if smoothen:
        spline = make_interp_spline(x, y)
        finer_x = np.linspace(x[0], x[len(x)-1], x_steps)  # x is assumed to be         
        #sorted.
        smoothed_y= spline(finer_x)
        x,y = finer_x,smoothed_y

    fig, axs = plt.subplots(1,figsize=(12,8))
    axs.plot(x,y)
    for vertical_line in vertical_lines:
        axs.axvline(vertical_line,linestyle='--',color='red',linewidth='1')

def main():
    global f_real, alpha_real, N_particles
    f_max = 10
    fs = np.arange(f_max/N_particles,f_max+f_max/N_particles,
                   2*f_max/N_particles) 
    fs = np.concatenate((-fs,fs)) # Consider negative frequencies as well.
    prior = {}
    for f in fs:
        prior[f] = 1/N_particles # We consider a flat prior up to f_max.
    
    steps = 100
    f_real = 8 #random.random()*f_max
    '''
    final_off_distribution =  offline_estimation(prior.copy(),f_max,steps)[3]
    off_cdf = cumulative_distribution_function(final_off_distribution)
    smooth_and_plot(off_cdf,vertical_lines=[f_real,-f_real])
    print("Offline")
    '''
    final_adaptive_distribution =  adaptive_estimation(prior.copy(),steps)[3]
    adapt_cdf = cumulative_distribution_function(final_adaptive_distribution)
    smooth_and_plot(adapt_cdf,vertical_lines=[f_real,-f_real])
    print("Adaptive")

    print("n=%d; N=%d; f_max=%d; alpha=%.2f; 1d, SMC" 
              % (N_particles,steps,f_max,alpha_real))
    
main()