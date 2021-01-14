# -*- coding: utf-8 -*-
"""
Hamiltonian learning implementation for a simple precession example, using both
offline and adaptive Bayesian inference.

The qubit is assumed to be initialized at state |+> for each iteration, and to
evolve under H = f*sigma_z/2, where f is the parameter to be estimated.

The prior is chosen to have support over negative frequencies, covering the two
modes of the expected frequency distribution (at +f and -f for f!=0).

A sequential Monte Carlo approximation is used to represent the probability 
distributions, using Hamiltonian Monte Carlo mutation steps.

The final cumulative distribution function is plotted, along with vertical lines 
marking the two symmetric real frequencies.
"""

import sys, random, matplotlib.pyplot as plt
from autograd import grad, numpy as np
np.seterr(all='warn')

dim=1
total_HMC, accepted_HMC = 0, 0
total_MH, accepted_MH = 0, 0

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

def likelihood(outcome, test_f, t):
    '''
    Provides an estimate for the likelihood  P(D|test_f,t) of an x-spin 
    measurement at time t yielding a given result, given a test parameter for  
    the fixed form Hamiltonian. 
    
    Parameters
    ----------
    outcome: int
        The result of the measurement (with |+> mapped to 1, and |-> to 0).
    test_f: float
        The test precession frequency.
    t: float
        The evolution time between the initialization and the projection.
        
    Returns
    -------
    p: float
        The estimated probability of obtaining the input outcome. 
    '''
    p1 = simulate_1(test_f,t)
    if outcome==1:
        p = p1
    if outcome==0:
        p = 1-p1
    return p 

def target_U(outcome, test_f,t):
    '''
    Evaluates the target "energy" associated to the likelihood at a time t seen
    as a probability density, given a parameter for the fixed form Hamiltonian. 
    
    Parameters
    ----------
    outcome: int
        The result of the measurement (with |+> mapped to 1, and |-> to 0).
    particle: float
        The frequency to be used for the likelihood.
    t: float
        The evolution time between the initialization and the projection.
        
    Returns
    -------
    U: float
        The value of the "energy".
    '''
    l = likelihood(outcome, test_f, t)
    U = -np.log(l)
    return(U)

def U_gradient(outcome,test_f,t,autograd=False):
    '''
    Evaluates the derivative of the target "energy" associated to the likelihood 
    at a time t seen as a probability density, given a frequency for the fixed 
    form Hamiltonian. 
    
    Parameters
    ----------
    outcome: int
        The result of the measurement (with |+> mapped to 1, and |-> to 0).
    test_f: float
        The frequency to be used for the likelihood.
    t: float
        The evolution time between the initialization and the projection.
    autograd: bool, optional
        Whether to use automatic differenciation (Default is False).
        
    Returns
    -------
    DU: float
        The derivative of the "energy".
    '''
    if autograd: 
        DU_f = grad(target_U,1)
        DU = DU_f(outcome,test_f,t)
    else:
        if outcome==1:
            DU=t*np.sin(test_f*t/2)/np.cos(test_f*t/2)
        if outcome==0:
            DU=-t*np.sin(test_f*t/2)*np.cos(test_f*t/2)/(1-
                                                         np.cos(test_f*t/2)**2)
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
def metropolis_hastings_step(outcome, t, particle, s=1, factor=1,
                             left_constraint = -10,right_constraint=10): 
    '''
    Performs a Metropolis-Hastings mutation on a given particle, using a 
    gaussian function for the proposals.
    
    Parameters
    ----------
    outcome: int
        The result of the measurement (with |+> mapped to 1, and |-> to 0).
    t: float
        The time at which the current iteration's measurement was performed.
        This is relevant because the target function and its derivative are 
        time-dependent.
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
    p = likelihood(outcome,t,new_particle)*gaussian(particle,new_particle,
                                                    sigma)/ \
        (likelihood(outcome,t,particle)*gaussian(new_particle,particle,sigma))
    return new_particle,p

def simulate_dynamics(outcome,t, initial_momentum, initial_particle, m, L, eta,
                      left_constraint = -10, right_constraint=100):    
    '''
    Simulates Hamiltonian dynamics for a given particle, using leapfrog 
    integration.
    
    Parameters
    ----------
    outcome: int
        The result of the measurement (with |+> mapped to 1, and |-> to 0).
    t: float
        The time at which the current iteration's measurement was performed.
        This is relevant because the target function and its derivative are 
        time-dependent.
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
    DU = U_gradient(outcome,new_particle,t)
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
            DU = U_gradient(outcome,new_particle,t)
            new_particle = new_particle - eta*DU
    new_momentum = new_momentum - 0.5*eta*DU

    p = np.exp(target_U(outcome,initial_particle,t)-\
               target_U(outcome,new_particle,t)+\
                   initial_momentum**2/(2*m)-new_momentum**2/(2*m))
    
    '''
    if (p<0.1):
        sys.exit('p too small')
    '''
    
    return new_particle, p
        
first = True
def hamiltonian_MC_step(outcome, t, point, m=1, L=20, eta=10**-10, 
                        threshold=0.1):
    '''
    Performs a Hamiltonian Monte Carlo mutation on a given particle.
    
    Parameters
    ----------
    outcome: int
        The result of the measurement (with |+> mapped to 1, and |-> to 0).
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
        new_point, p = simulate_dynamics(outcome,t,initial_momentum,point,
                                         m,L,eta)
    else:
        p = 0
        
    # If the Hamiltonian Monte Carlo acceptance probability is too low,
    #a Metropolis-Hastings mutation will be perform instead.
    # This is meant to saufegard the termination of the program if the leapfrog
    #integration is too inaccurate for a given set of parameters and experiment
    #controls.
    if (p < threshold):
        MH = True
        new_point, p = metropolis_hastings_step(outcome,t,point,s=1/m)
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

def bayes_update(outcome, distribution, t):
    '''
    Updates a prior distribution according to the outcome of a measurement, 
    using Bayes' rule. 
    
    Parameters
    ----------
    outcome: int
        The result of the measurement (with |+> mapped to 1, and |-> to 0).
    distribution: dict
        , with (key,value):=(frequency particle,importance weight)
        The prior distribution (SMC approximation). When returning, it will 
        have been updated according to the provided experimental datum.
        
    t: float
        The time at which the measurement was performed, which characterizes 
        it. The reference (t=0) is taken as the time of initialization.
    '''
    global N_particles
    
    # Update the weights.
    for particle in distribution:
        distribution[particle] = likelihood(outcome,particle,t)
        
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
            mutated_particle = hamiltonian_MC_step(outcome,t,particle
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
    # We'll use evenly spaced times, fixed in advance.
    mean, stdev = SMCparameters(distribution)
    means, stdevs = [], []
    means.append(mean)
    stdevs.append(stdev) 
    
    ts = np.arange(1, steps+1)*increment
    for t in ts:
        outcome = measure(t)
        # Update the distribution: get the posterior of the current iteration, 
        #which is the prior for the next.
        bayes_update(outcome,distribution,t) 
        
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
        p1 += simulate_1(particle,time)*distribution[particle]
    p0 = 1-p1
        
    dist_0 = distribution.copy()
    dist_1 = distribution.copy()
    
    # Update the distribution assuming each of the possible outcomes.
    bayes_update(dist_0,time,0) 
    bayes_update(dist_1,time,1)
    
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
        outcome = measure(adaptive_t)
        # Update the distribution: get the posterior of the current iteration, 
        #which is the prior for the next.
        bayes_update(outcome,distribution,adaptive_t)
        
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
    '''
    Computes the cumulative distribution function for a given distribution.
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(frequency particle,importance weight)
        The prior distribution (SMC approximation).
        
    Returns
    -------
    points: [(float,float)]
        A list of (frequency particle, cumulative probability) tuples assigning 
        to each particle in the input distribution (by order of increasing 
        frequency) the probability of the frequency being equal to or below its 
        own, according to the input distribution.
    '''
    sorted_particles = sorted([key for key in list(distribution.keys())])
    cumulative_sum = 0
    points=[]
    for particle in sorted_particles:
        cumulative_sum += 1/N_particles # The particles have uniform weights  
        #after each iteration
        points.append((particle,cumulative_sum))
    return points

def plot_tuples(points, vertical_lines=[]):
    '''
    Plots the curve corresponding to a list of 2 element tuples, and 
    (optionally) dashed vertical lines for a specified set of x-coordinates.
    
    Parameters
    ----------
    points: [(float,float)]
        A list of tuples, sorted by the first elements.
    '''
    x,y = [[point[i] for point in points] for i in range(2)]

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
    plot_tuples(off_cdf,vertical_lines=[f_real,-f_real])
    print("Offline")
    '''
    final_adaptive_distribution =  adaptive_estimation(prior.copy(),steps)[3]
    adapt_cdf = cumulative_distribution_function(final_adaptive_distribution)
    plot_tuples(adapt_cdf,vertical_lines=[f_real,-f_real])
    print("Adaptive")

    print("n=%d; N=%d; f_max=%d; alpha=%.2f; 1d, SHMC" 
          % (N_particles,steps,f_max,alpha_real))
    
    global total_HMC, accepted_HMC, total_MH, accepted_MH
    print("* Percentage of HMC steps:  %.1f%%." 
          % (100*total_HMC/(total_HMC+total_MH)))
    if (total_HMC != 0):
        print("* Hamiltonian Monte Carlo: %d%% mean particle acceptance rate." 
              % round(100*accepted_HMC/total_HMC))
    if (total_MH != 0):
        print("* Metropolis-Hastings:     %d%% mean particle acceptance rate." 
              % round(100*accepted_MH/total_MH))
    
main()