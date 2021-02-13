# -*- coding: utf-8 -*-
"""
Hamiltonian learning implementation for a simple precession example, using both
offline and adaptive Bayesian inference.

The qubit is assumed to be initialized at state |+> for each iteration, and to
evolve under H = f*sigma_z/2, where f is the parameter to be estimated.

A sequential Monte Carlo approximation is used to represent the probability 
distributions, using Hamiltonian Monte Carlo and Metropolis-Hastings mutation 
steps.

The evolution of the standard deviations the "precisions" (the variance times 
the cumulative evolution time) with the steps are plotted, and the final values 
of these quantities (in addition to the actual error) are printed.

The algorithm is repeated for a number of runs with randomly picked real
values, and medians are taken over all of them to get the results and graphs.
"""

import sys, random, matplotlib.pyplot as plt
from autograd import grad, numpy as np
np.seterr(all='warn')

dim=1
total_HMC, accepted_HMC = 0, 0
total_MH, accepted_MH = 0, 0

N_particles = 100 # Number of samples used to represent the probability
#distribution, using a sequential Monte Carlo approximation.

f_real = 0 # The actual precession frequency we mean to estimate 
#(will be picked at random for each run).

alpha_real = 0 # The decay factor (the inverse of the coherence time)
#(will be picked at random for each run).

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

def simulate_1(test_f, t):
    '''
    Provides an estimate for the likelihood  P(D=1|test_f,t) of an x-spin 
    measurement at time t yielding result |+>, given a test parameter for the 
    fixed form Hamiltonian. 
    
    Parameters
    ----------
    test_f: float
        The test precession frequency.
    t: float
        The evolution time between the initialization and the projection.
        
    Returns
    -------
    p1: float
        The estimated probability of finding the particle at state |+>.
    '''
    p1 = np.cos(test_f*t/2)**2
    return p1

def likelihood(data, test_f):
    '''
    Provides an estimate for the likelihood  P(D|test_f,t) of x-spin 
    measurements yielding the input vector of data, given a test parameter for  
    the fixed form Hamiltonian. 
    
    Parameters
    ----------
    data: [(float,int)]
        A vector of experimental results and controls, each datum being of the 
        form (time,outcome), where 'time' is the control used for each 
        experiment and 'outcome' is its result.
    test_f: float
        The test precession frequency.
        
    Returns
    -------
    p: float
        The estimated probability of obtaining the input outcome. 
    '''

    if np.size(data)==2: # Single datum case.
        t,outcome = data
        p = simulate_1(test_f,t)*(outcome==1)+\
            (1-simulate_1(test_f,t))*(outcome==0) 
    else:
        p = np.product([likelihood(datum, test_f) for datum in data])
    return p 

def loglikelihood(data, test_f):
    '''
    Evaluates the log-likelihood (symmetric of the target "energy") associated 
    to the joint likelihood of some vector  of data, given a test parameter for 
    the fixed form Hamiltonian. 
    
    Parameters
    ----------
    data: [(float,int)]
        A vector of experimental results and controls, each datum being of the 
        form (time,outcome), where 'time' is the control used for each 
        experiment and 'outcome' is its result.
    test_f: float
        The test precession frequency.
        
    Returns
    -------
    p: float
        The estimated probability of obtaining the input outcome. 
    '''
    p = np.sum([np.log(likelihood(datum, test_f)) for datum in data])
    return p 

def U_gradient(data,test_f,autograd=False):
    '''
    Evaluates the derivative of the target "energy" associated to the likelihood 
    at a time t seen as a probability density, given a frequency for the fixed 
    form Hamiltonian. 
    
    Parameters
    ----------
    data: [(float,int)]
        A vector of experimental results and controls, each datum being of the 
        form (time,outcome), where 'time' is the control used for each 
        experiment and 'outcome' is its result.
    test_f: float
        The frequency to be used for the likelihood.
    autograd: bool, optional
        Whether to use automatic differenciation (Default is False).
        
    Returns
    -------
    DU: float
        The derivative of the "energy".
    '''
    
    '''
    The cosine must be different from 0 if the outcome is 1, and from -1/1 if 
    the outcome is 0 (so that the divisor is non-zero when evaluating the            
    derivative).
    '''
    usable_data = [(t,outcome) for (t,outcome) in data 
            if (np.cos(test_f*t/2)!=1 or outcome!=1)
            and (abs(np.cos(test_f*t/2))!=1 or outcome!=0)]
    
    if autograd: 
        minus_DU_f = grad(loglikelihood,1)
        DU = -minus_DU_f(usable_data,float(test_f))
    else:
        DU = 0
        for (t,outcome) in usable_data:
            if outcome==1:
                DU+=t*np.sin(test_f*t/2)/np.cos(test_f*t/2)
            if outcome==0:
                DU+=-t*np.sin(test_f*t/2)*np.cos(test_f*t/2)\
                    /np.sin(test_f*t/2)**2
    return(DU)

def gaussian(x, mu, sigma, normalize=False):
    '''
    Evaluates a gaussian function at a given point for some specified set of
    parameters.
    
    Parameters
    ----------
    x: float
        The point at which the function is to be evaluated.
    mu: float
        The mean to be used for the gaussian function.
    sigma: float
        The standard deviation to be used for the gaussian function.
    normalize: bool, optional
        Whether to normalize the result (Default is False).
        
    Returns
    -------
    e: float
        The value of the gaussian function at the provided point.
    '''
    power = -(x-mu)**2/(2*sigma**2)
    if not normalize:
        e = np.exp(power)
        return e
    else:
        norm = (2*np.pi*sigma**2)**0.5  
        e = e/norm
        return e

first_metropolis_hastings_step = True
def metropolis_hastings_step(data, particle, s=1, factor=1,
                             left_constraint = 0,right_constraint=10): 
    '''
    Performs a Metropolis-Hastings mutation on a given particle, using a 
    gaussian function for the proposals.
    
    Parameters
    ----------
    data: [(float,int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    particle: float
        The particle to undergo a mutation step.
    s: float, optional
        The standard deviation to be multiplied by a factor and then used as 
        standard deviation for the normal distribution used for the proposal 
        (Default is 1).
    factor: float, optional
        The factor 's' should be be multiplied by to get the standard deviation
        of the the normal distribution used for the proposal (Default is 0.05).
    left_constraint: float
        The leftmost bounds to be enforced for the particle's motion.
    right_constraint: float
        The rightmost bounds to be enforced for the particle's motion.
        
    Returns
    -------
    particle: float
        The mutated particle.
    p: float
        The acceptance probability to be used for the evolved particle as a 
        Monte Carlo proposal.
    '''
    global first_metropolis_hastings_step
    if first_metropolis_hastings_step:
      if (s!=1):
          cov = "cov^-1"
      else:
          cov = "1"
      print("MH:  s=%s, factor=%.4f" % (cov,factor))
      first_metropolis_hastings_step = False

    sigma = s*factor

    # Start with any invalid value.
    new_particle = left_constraint-1
    
    # Get a proposal that satisfies the constraints.
    while (new_particle < left_constraint or new_particle > right_constraint):
        new_particle = np.random.normal(particle, sigma)
        
    # Compute the probabilities of transition for the acceptance probability.
    p = likelihood(data,new_particle)*gaussian(particle,new_particle,
                                                    sigma)/ \
        (likelihood(data,particle)*gaussian(new_particle,particle,sigma))
    return new_particle,p

def simulate_dynamics(data, initial_momentum, initial_particle, m, L, eta,
                      left_constraint = 0, right_constraint=10):    
    '''
    Simulates Hamiltonian dynamics for a given particle, using leapfrog 
    integration.
    
    Parameters
    ----------
    data: [(float,int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    initial_momentum: float
        The starting momentum vector. 
    initial_particle: float
        The particle for which the Hamiltonian dynamics is to be simulated.
    m: float
        The mass to be used when simulating the Hamiltonian dynamics (a HMC 
        tuning parameter).
    L: int
        The amount of integration steps to be used when simulating the 
        Hamiltonian dynamics (a HMC tuning parameter).
    eta: float
        The integration stepsize to be used when simulating the Hamiltonian 
        dynamics (a HMC tuning parameter).
    left_constraints: float
        The leftmost bound to be enforced for the particle's motion.
    right_constraints: float
        The rightmost bound to be enforced for the particle's motion.
        
    Returns
    -------
    particle: float
        The particle having undergone motion.
    p: float
        The acceptance probability to be used for the evolved particle as a 
        Monte Carlo proposal.
    '''    
    new_particle = initial_particle
    DU = U_gradient(data,new_particle)
    global f_real
    
    # Perform leapfrog integration according to Hamilton's equations.
    new_momentum = initial_momentum - 0.5*eta*DU
    for l in range(L):
        new_particle = new_particle + eta*new_momentum/m
        
        # Enforce the constraint that both the frequency lie within the prior 
        #distribution. 
        # Should a limit be crossed, the position and momentum are chosen such 
        #that the particle "rebounds".
        if (new_particle < left_constraint):
            new_particle = left_constraint+(left_constraint-new_particle)
            new_momentum = -new_momentum
        if (new_particle > right_constraint): # Use the upper limit from the 
        #prior.
            new_particle = right_constraint-(new_particle-right_constraint)
            new_momentum = -new_momentum
        if (l != L-1):
            DU = U_gradient(data,new_particle)
            new_particle = new_particle - eta*DU
    new_momentum = new_momentum - 0.5*eta*DU


    p = np.exp(-loglikelihood(data,initial_particle)-\
               (-loglikelihood(data,new_particle))+\
                   initial_momentum**2/(2*m)-new_momentum**2/(2*m))
    
    '''
    if (p<0.1):
        sys.exit('p too small')
    '''
    
    return new_particle, p
        
first = True
def hamiltonian_MC_step(data, point, m=1, L=20, eta=10**-7, 
                        threshold=0.1):
    '''
    Performs a Hamiltonian Monte Carlo mutation on a given particle.
    
    Parameters
    ----------
    data: [(float,int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    t: float
        The time at which the current iteration's measurement was performed.
        This is relevant because the target function and its derivative are 
        time-dependent.
    particle: float
        The frequency particle to undergo a mutation step.
    m: float, optional
        The mass to be used when simulating the Hamiltonian dynamics (a HMC 
        tuning parameter) (Default is 1).
    L: int, optional
        The amount of integration steps to be used when simulating the 
        Hamiltonian dynamics (a HMC tuning parameter) (Default is 20).
    eta: float, optional
        The integration stepsize to be used when simulating the Hamiltonian 
        dynamics (a HMC tuning parameter) (Default is exp(-4)).
    threshold: float, optional
        The highest HMC acceptance rate that should trigger a Metropolis-
        -Hastings mutation step (as an alternative to a  HMC mutation step) 
        (Default is 0.1). 
        
    Returns
    -------
    particle: float
        The mutated frequency particle.
    '''
    if (threshold<1):
        # Perform a Hamiltonian Monte Carlo mutation.
        global first
        if first:
            if (m!=1):
                mass = "cov"
            else:
                mass = "1"
            print("HMC: m=%s, L=%d, eta=%f" % (mass,L,eta))
            first = False
            
        global total_HMC, accepted_HMC, total_MH, accepted_MH
        initial_momentum = np.random.normal(0, scale=m)
        new_point, p = simulate_dynamics(data,initial_momentum,point,
                                         m,L,eta)
    else:
        p = 0
        
    # If the Hamiltonian Monte Carlo acceptance probability is too low,
    #a Metropolis-Hastings mutation will be performed instead.
    # This is meant to saufegard the termination of the program if the leapfrog
    #integration is too inaccurate for a given set of parameters and experiment
    #controls (which tends to happen close to or at the assymptotes of the log-
    #-likelihood).
    
    if (p < threshold):
        MH = True
        new_point, p = metropolis_hastings_step(data,point,s=1/m)
        total_MH += 1
    else:
        MH = False
        total_HMC += 1
        
    a = min(1,p)
    if (np.random.rand() < a):
        if MH:
            accepted_MH += 1
        else:
            accepted_HMC += 1
        return(new_point)
    else:
        return(point)

def bayes_update(data, distribution):
    '''
    Updates a prior distribution according to the outcome of a measurement, 
    using Bayes' rule. 
    
    Parameters
    ----------
    data: [(float,int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    distribution: dict
        , with (key,value):=(frequency particle,importance weight)
        The prior distribution (SMC approximation). When returning, it will 
        have been updated according to the provided experimental datum.
        
    t: float
        The time at which the measurement was performed, which characterizes 
        it. The reference (t=0) is taken as the time of initialization.
    '''
    global N_particles
    
    # Update weights based on last likelihood obtained (as in SMC). 
    for particle in distribution:
        distribution[particle] = likelihood(data[-1],particle)
        
    selected_particles = random.choices(list(distribution.keys()), 
                                          weights=distribution.values(),
                                          k=N_particles)
    
    cov = SMCparameters(selected_particles)[1]**2
    # The covariance will be used as a mass, so it must be positive.
    if (cov==0):
        return
    
    distribution.clear()
    for particle in selected_particles:
        repeated = True
        while (repeated == True):
            mutated_particle = hamiltonian_MC_step(data,particle
                                                   ,m=1/cov)
            if (mutated_particle not in distribution):
                repeated = False
        distribution[mutated_particle] = 1

def SMCparameters(distribution, stdev=True):
    '''
    Calculates the mean and (optionally) standard deviation of a given 
    distribution.
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(frequency particle,importance weight)
        The distribution (SMC approximation) whose parameters are to be 
        calculated. This can also be a list if the weights are uniform.
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
        mean += f
        meansquare += f**2
    mean = mean/N_particles
    meansquare = meansquare/N_particles
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
    mean, stdev = SMCparameters(distribution)
    means, stdevs = [], []
    means.append(mean)
    stdevs.append(stdev) 
    data = []
    # We'll use evenly spaced times, fixed in advance.
    ts = np.arange(1, steps+1)*increment
    #ts = np.arange(1, steps+1)*increment
    for t in ts:
        data.append((t,measure(t)))
        # Update the distribution: get the posterior of the current iteration, 
        #which is the prior for the next.
        bayes_update(data,distribution) 
        
        mean,stdev = SMCparameters(distribution)
        means.append(mean)
        stdevs.append(stdev) 
    cumulative_times = np.cumsum([i/(2*f_max) for i in range(steps+1)])
        
    return means, stdevs, cumulative_times
    
def expected_utility(data, distribution, time):
    '''
    Returns the expectation value for the utility of a measurement time.
    The utility function considered is a weighed sum of the negative variances
    of the two parameters, adjusted for their relative scale.
    Its expectation value is computed over all outcomes, given the current
    distribution.
    
    Parameters
    ----------
    data: [(float,int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
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
        p1 += likelihood([(1,time)],particle)
    p0 = 1-p1
        
    dist_0 = distribution.copy()
    dist_1 = distribution.copy()
    
    # Update the distribution assuming each of the possible outcomes.
    bayes_update(data+[(time,0)],dist_0) 
    bayes_update(data+[(time,1)],dist_1)
    
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
        chosen.
        
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

first_adaptive_estimation = True
def adaptive_estimation(distribution, steps, k=1, guesses=1,precision=0):
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
    steps: int
        The maximum number of experiments to be performed.
    k: float, optional
        The proportionality constant to be used for the particle guess 
        heuristic (Default is 1.25).
    guesses: int, optional
        The amount of hypothesis to be picked for the time; only the one which      
        maximizes the expected utility among this set will be chosen (Default 
        is 1).
        If this quantity is greater than one, the times will be chosen to be 
        inversely proportional to the distance between two particles picked at
        random from the current distribution (instead of to its standard 
        deviation), in order to introduce variability.
    precision: float, optional
        The threshold precision required to stop the learning process before  
        attaining the step number limit (Default is 0).
        
        
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
    global first_adaptive_estimation
    if first_adaptive_estimation is True:
        print("Adaptive estimation: k=%.2f; %d guess(es) per step" % 
              (k,guesses))
        first_adaptive_estimation = False
        
    mean, stdev = SMCparameters(distribution)
    means, stdevs, cumulative_times = [], [], []
    means.append(mean)
    stdevs.append(stdev)
    cumulative_times.append(0)
    data = []
    
    if (guesses==1):
        adaptive_t = k/stdev
    else:
        adaptive_t = adaptive_guess(distribution,k,guesses)
        
    cumulative_times.append(adaptive_t)
        
    for i in range(1,steps+1):
        data.append((adaptive_t,measure(adaptive_t)))
        # Update the distribution: get the posterior of the current iteration, 
        #which is the prior for the next.
        bayes_update(data,distribution)
        
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
            
    return means, stdevs, cumulative_times

def main():
    global f_real, alpha_real, N_particles
    f_max = 10
    fs = np.arange(f_max/N_particles,f_max+f_max/N_particles,
                   f_max/N_particles) 
    prior = {}
    for f in fs:
        prior[f] = 1/N_particles # We consider a flat prior up to f_max.
    
    runs=1
    steps = 30
    adapt_runs, off_runs = [], []
    adapt_errors, off_errors = [], []
    adapt_mses, off_mses = [], []
    
    for i in range(runs):
        f_real = random.random()*f_max
        adapt_runs.append(adaptive_estimation(prior.copy(),steps,precision=0))
        off_runs.append(offline_estimation(prior.copy(),f_max,steps))
        
        adapt_errors.append(abs(adapt_runs[i][0][steps-1]-f_real))
        adapt_mses.append((adapt_runs[i][0][steps-1]-f_real)**2)
        off_errors.append(abs(off_runs[i][0][steps-1]-f_real))
        off_mses.append((off_runs[i][0][steps-1]-f_real)**2)
        
        if (i%(runs/10)==0):
            sys.stdout.write('.');sys.stdout.flush();
    '''
    The indexes in adapt_runs/off_runs are, by order: 
        - Run number;
        - Desired quantity (0 for mean, 1 for stdev, 2 for cumulative_time)
        - Step number
    '''

    adapt_stdevs = [np.percentile([s[i] for m,s,t in adapt_runs],
                                  50,interpolation='nearest') \
                   for i in range(steps+1)]
        
    ''' Use percentile with "nearest" interpolation for the standard deviation 
    median so that the presented value actually corresponds to a run, because
    it's more relevant to know the error corresponding to that specific run
    than its median over all runs (to see how well the uncertainty covers the
    error for a fixed run, rather than the independent medians for errors and
    standard deviations which may be associated to different runs).
    
    We then get the error from the selected run (the one corresponding to the 
    median of all final - i.e. last step - standard deviation values):
    '''
        
    final_adapt_stdevs = [s[steps] for m,s,t in adapt_runs]
    median_final_adapt_stdevs_index = final_adapt_stdevs.index(
        adapt_stdevs[steps])
    adapt_error = adapt_errors[median_final_adapt_stdevs_index]
    adapt_mse = adapt_mses[median_final_adapt_stdevs_index]

    adapt_times = [np.median([t[i] for m,s,t in adapt_runs]) \
                   for i in range(steps+1)]
        
    adapt_precisions_all = [([s[i]**2*t[i] for i in range(steps+1)]) \
                   for m,s,t in adapt_runs]
    adapt_precisions = [np.median([adapt_precisions_all[i][j] \
                      for i in range(runs)]) for j in range(steps+1)]
    
    adapt_stdevs_q1s = [np.percentile([s[i] for m,s,t in adapt_runs], 25) \
                        for i in range(steps+1)]
    adapt_stdevs_q3s = [np.percentile([s[i] for m,s,t in adapt_runs], 75) \
                        for i in range(steps+1)]
    
    adapt_precisions_q1s = [np.percentile([adapt_precisions_all[i][j] \
                                           for i in range(runs)], 25) \
                            for j in range(steps+1)]
    adapt_precisions_q3s = [np.percentile([adapt_precisions_all[i][j] \
                                           for i in range(runs)], 75) \
                            for j in range(steps+1)]
    
    print("\nAdaptive:\n- Variance: %.4f\n- MSE: %.4f (actual error %.2f)\n"\
          "- Final precision: %.2f" % (adapt_stdevs[steps]**2, 
                                       adapt_mse,
                                       adapt_error, 
                                       adapt_precisions[steps]))
        
    # Do the same thing as before for the standard deviations and errors.
    off_stdevs = [np.percentile([s[i] for m,s,t in off_runs],
                                  50,interpolation='nearest') \
                   for i in range(steps+1)]
        
    final_off_stdevs = [s[steps] for m,s,t in off_runs]
    median_final_off_stdevs_index = final_off_stdevs.index(off_stdevs[steps])
    off_error = off_errors[median_final_off_stdevs_index]
    off_mse = off_mses[median_final_off_stdevs_index]
        
    off_times = [np.median([t[i] for m,s,t in off_runs]) \
                   for i in range(steps+1)]
    off_precisions_all = [([s[i]**2*t[i] for i in range(steps+1)]) \
                   for m,s,t in off_runs]
    off_precisions = [np.median([off_precisions_all[i][j] \
                      for i in range(runs)]) for j in range(steps+1)]
    
    off_stdevs_q1s = [np.percentile([s[i] for m,s,t in off_runs], 25) \
                        for i in range(steps+1)]
    off_stdevs_q3s = [np.percentile([s[i] for m,s,t in off_runs], 75) \
                        for i in range(steps+1)]
    off_precisions_q1s = [np.percentile([off_precisions_all[i][j] \
                                           for i in range(runs)], 25) \
                            for j in range(steps+1)]
    off_precisions_q3s = [np.percentile([off_precisions_all[i][j] \
                                           for i in range(runs)], 75) \
                            for j in range(steps+1)]
    
    print("Offline:\n- Variance: %.4f\n- MSE: %.4f (actual error %.2f)\n"\
          "- Final precision: %.2f" % (off_stdevs[steps]**2, 
                                       off_mse,
                                       off_error,
                                       off_precisions[steps]))
        
    print("(n=%d; N=%d; f_max=%d; alpha=%.2f; runs=%d; 1d, SHMC)" 
          % (N_particles,steps,f_max,alpha_real,runs))
    
    global total_HMC, accepted_HMC, total_MH, accepted_MH
    print("* Percentage of HMC steps:  %.1f%%." 
          % (100*total_HMC/(total_HMC+total_MH)))
    if (total_HMC != 0):
        print("* Hamiltonian Monte Carlo: %d%% mean particle acceptance rate." 
              % round(100*accepted_HMC/total_HMC))
    if (total_MH != 0):
        print("* Metropolis-Hastings:     %d%% mean particle acceptance rate." 
              % round(100*accepted_MH/total_MH))
        
    fig, axs = plt.subplots(2,figsize=(8,8))
    
    x1 = np.array([i for i in range(steps+1)])
    y1 = np.array([adapt_stdevs[i] for i in range(steps+1)])
    oy1 = np.array([off_stdevs[i] for i in range(steps+1)])
    axs[0].set_ylabel(r'$\sigma$')
    axs[0].plot(x1, y1, label='adaptive')
    axs[0].plot(x1, oy1, color='red', label='offline')
    q11 = np.array([adapt_stdevs_q1s[i] for i in range(steps+1)])
    q31 = np.array([adapt_stdevs_q3s[i] for i in range(steps+1)])
    oq11 = np.array([off_stdevs_q1s[i] for i in range(steps+1)])
    oq31 = np.array([off_stdevs_q3s[i] for i in range(steps+1)])
    axs[0].fill_between(x1,q11,q31,alpha=0.1)
    axs[0].fill_between(x1,oq11,oq31,alpha=0.1,color='red')
    axs[0].set_title('Adaptive vs. Offline Estimation')
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[0].set_xlabel('Iteration number')
    
    x2 = np.array([adapt_times[i] for i in range(1,steps+1)])
    y2 = np.array([adapt_precisions[i] for i in range(1,steps+1)])
    ox2 = np.array([off_times[i] for i in range(1,steps+1)])
    oy2 = np.array([off_precisions[i] for i in range(1,steps+1)])
    q12 = np.array([adapt_precisions_q1s[i] for i in range(1,steps+1)])
    q32 = np.array([adapt_precisions_q3s[i] for i in range(1,steps+1)])
    oq12 = np.array([off_precisions_q1s[i] for i in range(1,steps+1)])
    oq32 = np.array([off_precisions_q3s[i] for i in range(1,steps+1)])
    axs[1].set_ylabel(r'$\sigma^2 \cdot \Delta t$')
    axs[1].loglog(x2, y2, label='adaptive')
    axs[1].loglog(ox2, oy2,color='red', label='offline')
    axs[1].fill_between(x2,q12,q32,alpha=0.1)
    axs[1].fill_between(ox2,oq12,oq32,alpha=0.1,color='red')
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[1].set_xlabel('Total accumulation time')
    
    fig.subplots_adjust(hspace=0.5)
    
main()