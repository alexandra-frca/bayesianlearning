# -*- coding: utf-8 -*-
"""
Hamiltonian learning implementation for a simple precession example, using both
offline and adaptive Bayesian inference.

The qubit is assumed to be initialized at state |+> for each iteration, and to
evolve under H = f*sigma_z/2 (where f is the parameter to be estimated), apart
from the exponential decay resulting from the presence of decoherence.

A sequential Monte Carlo approximation is used to represent the probability 
distributions, along with Liu-West resampling.
"""

import sys, random, numpy as np, matplotlib.pyplot as plt

N_particles = 2500 # Number of samples used to represent the probability
#distribution, using a sequential Monte Carlo approximation.

parameters = {}

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
        The exponential decay parameter (Default is alpha_real).
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

def simulate(test_f, test_alpha, t, runs=500):
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
    test_alpha: float
        The test decay factor (the inverse of the coherence time).
    t: float
        The evolution time between the initialization and the projection.
    runs: int, optional
        The total number of measurements performed on the simulated system, 
        which will determine the accuracy of the estimate (Default is 500).
        
    Returns
    -------
    p: float
        The estimated probability of finding the particle at state |+>.
    '''
    p = np.random.binomial(runs,
                           p=(np.cos(test_f*t/2)**2*np.exp(-test_alpha*t)+
                            (1-np.exp(-test_alpha*t))/2))/runs
    return p 

def resample(particle_f, particle_alpha, distribution, resampled_distribution, 
             a=0.85):
    '''
    Resamples a particle from a given amount of times and adds the results to 
    a previous distribution. Uniform weights are attributed to the resampled 
    particles; the distribution is caracterized by particle density.
    For a=0, the resampling works as a bootstrap filter.
    
    Parameters
    ----------
    f_particle: float
        The frequency of the particle to be resampled.
    f_particle: float
        The decay factor (the inverse of the coherence time) of the particle 
        to be resampled.
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle=(f,alpha):=(frequency,decay factor)
        The prior distribution (SMC approximation).
    resampled_distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle=(f,alpha):=(frequency,decay factor)
        The distribution (SMC approximation) in the process of being resampled. 
        When returning, it will have been updated with the freshly resampled 
        particle(s).
    a: float, optional
        The Liu-West filtering parameter (Default is 0.85).
    '''
    current_means, current_stdevs = SMCparameters(distribution)
    current_f_mean, current_alpha_mean = current_means
    current_f_stdev, current_alpha_stdev = current_stdevs
    # Sample a positive frequency particle.
    new_particle=(-1,-1)
    while (new_particle[0]==(-1,-1)):
        new_f = np.random.normal(a*particle_f+(1-a)*current_f_mean,
                                      scale=(1-a**2)**0.5*current_f_stdev)
        new_alpha = np.random.normal(a*particle_alpha+(1-a)*current_alpha_mean,
                                      scale=(1-a**2)**0.5*current_alpha_stdev)
        new_particle = (new_f,new_alpha)
        # Avoid repeated particles for variety.
        if new_particle in resampled_distribution:
            new_particle=(-1,-1)
    # Attribute uniform weights to the resampled particles.
    resampled_distribution[new_particle] = 1/N_particles

def bayes_update(distribution, t, outcome, threshold=N_particles/2):
    '''
    Updates a prior distribution according to the outcome of a measurement, 
    using Bayes' rule. 
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle=(f,alpha):=(frequency,decay factor)
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
        p1 = simulate(particle[0],particle[1],t)
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
        resampled_distribution = {}
        for i in range(N_particles):
            particle = random.choices(list(distribution.keys()), 
                                  weights=distribution.values())[0]
            resample(particle[0],particle[1],distribution,
                     resampled_distribution)
        distribution = resampled_distribution


def SMCparameters(distribution, stdev=True):
    '''
    Calculates the mean and (optionally) standard deviation of a given 
    distribution.
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle=(f,alpha):=(frequency,decay factor)
        The distribution (SMC approximation) whose parameters are to be 
        calculated.
    stdev: bool
        To be set to False if the standard deviation is not to be returned 
        (Default is True).
        
    Returns
    -------
    means: (float,float)
        The means of the distribution along its two dimensions: frequency and
        decay factor (the inverse of the coherence time), by this order.
    stdevs: (float,float)
        The standard deviation of the distribution along its two dimensions: 
        frequency and decay factor (the inverse of the coherence time), by this 
        order.
    '''
    f_mean = 0
    f_meansquare = 0
    alpha_mean = 0
    alpha_meansquare = 0
    for particle in distribution:
        f = float(particle[0])
        alpha = float(particle[1])
        w = distribution[particle]
        f_mean += f*w
        f_meansquare += f**2*w
        alpha_mean += alpha*w
        alpha_meansquare += alpha**2*w
    means = (f_mean,alpha_mean)
    if not stdev:
        return means
    f_stdev = abs(f_mean**2-f_meansquare)**0.5
    alpha_stdev = abs(alpha_mean**2-alpha_meansquare)**0.5
    stdevs = (f_stdev,alpha_stdev)
    return means,stdevs

def offline_estimation(distribution, f_max, steps, increment=0.08):
    '''
    Estimates the precession frequency by defining a set of experiments, 
    performing them, and updating a given prior distribution according to their 
    outcomes (using Bayesian inference).
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle=(f,alpha):=(frequency,decay factor)
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
    current_mean, current_stdev = SMCparameters(distribution)
    means, stdevs = [], []
    means.append(current_mean)
    stdevs.append(current_stdev) 
    
    ts = np.arange(1, steps+1)*increment
    for t in ts:
        m = measure(t)
        # Update the distribution: get the posterior of the current iteration, 
        #which is the prior for the next.
        bayes_update(distribution,t,m) 
        
        current_mean, current_stdev = SMCparameters(distribution)
        means.append(current_mean)
        stdevs.append(current_stdev) 
        
    cumulative_times = np.cumsum([i/(2*f_max) for i in range(steps+1)])
    return means, stdevs, cumulative_times

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
        , and particle=(f,alpha):=(frequency,decay factor)
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
        p1 += simulate(float(particle[0]),float(particle[1]),time)\
            *distribution[particle]
    p0 = 1-p1
        
    dist_0 = distribution.copy()
    dist_1 = distribution.copy()
    
    # Update the distribution assuming each of the possible outcomes.
    bayes_update(dist_0,time,0) 
    bayes_update(dist_1,time,1)
    
    # Compute the expected utility for each oucome.
    stdevs_0 = SMCparameters(dist_0)[1]
    stdevs_1 = SMCparameters(dist_1)[1]
    util_0 = -(stdevs_0[0]**2+(100*stdevs_0[1])**2)
    util_1 = -(stdevs_1[0]**2+(100*stdevs_1[1])**2)
    
    # Calculate the expected utility over all (both) outcomes.
    utility = p0*util_0 + p1*util_1
    
    return(utility)

def adaptive_guess(distribution, k, guesses=5):
    '''
    Provides a guess for the evolution time to be used for a measurement,
    picked using the PGH, a particle guess heuristic (where the times are 
    chosen to be inverse to the distance between two particles sampled at 
    random from the distribution).
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle=(f,alpha):=(frequency,decay factor)
        The prior distribution (SMC approximation).
    k: float
        The proportionality constant to be used for the particle guess 
        heuristic.
    guesses: int
        The amount of hypothesis to be picked for the time using the PGH; only 
        the one which maximizes the expected utility among this set will be  
        chosen (Default is 5).
        
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
            delta = ((float(f1[0])-float(f2[0]))**2 + \
                     (100*(float(f1[1])-float(f2[1])))**2)**0.5
        adaptive_ts.append(k/delta)
    for t in adaptive_ts:
        utilities.append(expected_utility(distribution, t))
        
    return(adaptive_ts[np.argmax(utilities)])
    
def adaptive_estimation(distribution, steps, precision=0, k=3.5):
    '''
    Estimates the precession frequency by adaptively performing a set of 
    experiments, using the outcome of each to update the prior distribution 
    (using Bayesian inference) as well as to decide the next experiment to be 
    performed.
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle=(f,alpha):=(frequency,decay factor)
        The prior distribution (SMC approximation).
    steps: int, optional
        The maximum number of experiments to be performed.
    precision: float
        The threshold precision required to stop the learning process before  
        attaining the step number limit (Default is 0).
    k: float
        The proportionality constant to be used for the particle guess 
        heuristic (Default is 3.5).
        
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
    current_means, current_stdevs = SMCparameters(distribution)
    means, stdevs, cumulative_times = [], [], []
    means.append(current_means)
    stdevs.append(current_stdevs) 
    cumulative_times.append(0)
    
    adaptive_t = adaptive_guess(distribution,k)
    cumulative_times.append(adaptive_t)

    for i in range(1,steps+1):
        m = measure(adaptive_t)
        # Update the distribution: get the posterior of the current iteration, 
        #which is the prior for the next.
        bayes_update(distribution,adaptive_t,m)
        
        current_means, current_stdevs = SMCparameters(distribution)
        means.append(current_means)
        stdevs.append(current_stdevs) 
        
        if (current_stdevs[0]<precision and current_stdevs[1]<precision/100): 
            for j in range(i+1,steps+1): # Fill in the rest to get fixed length
        #lists for the plots.
                means.append(current_means)
                stdevs.append(current_stdevs)
                cumulative_times.append(cumulative_times[i-1])
            break
            
        adaptive_t = adaptive_guess(distribution,k)
        cumulative_times.append(adaptive_t+cumulative_times[i-1])      
    return means, stdevs, cumulative_times

def main():
    global f_real, alpha_real, N_particles
    f_max = 10
    alpha_max = 0.1
    f_particles = N_particles**0.5
    alpha_particles = N_particles/f_particles
    fs = np.arange(f_max/f_particles,f_max,f_max/(f_particles+1)) 
    alphas = np.arange(0,alpha_max,alpha_max/alpha_particles) 
    prior = {}
    for f in fs:
        for alpha in alphas:
            prior[(f,alpha)] = 1/N_particles
    
    runs=5
    steps = 50
    adapt_runs, off_runs = [], []
    adapt_errors, off_errors = [[],[]], [[],[]]
    
    
    parameters['f_real'] = [] 
    parameters['alpha_real'] = []  
    for i in range(runs):
        parameters['f_real'].append(random.random()*f_max)
        parameters['alpha_real'].append(random.random()*alpha_max)
    
    for i in range(runs):
        if (i%(runs/10)==0):
            sys.stdout.write('.');sys.stdout.flush();
        f_real = parameters['f_real'][i]
        alpha_real = parameters['alpha_real'][i]
        adapt_runs.append(adaptive_estimation(prior.copy(),steps,precision=0))
        off_runs.append(offline_estimation(prior.copy(),f_max,steps))
        adapt_errors[0].append(abs(adapt_runs[i][0][steps-1][0]-f_real))
        adapt_errors[1].append(abs(adapt_runs[i][0][steps-1][1]-alpha_real))
        off_errors[0].append(abs(off_runs[i][0][steps-1][0]-f_real))
        off_errors[1].append(abs(off_runs[i][1][steps-1][1]-alpha_real))
        
    adapt_times = [np.median([t[i] for m,s,t in adapt_runs]) \
         for i in range(steps+1)]
    adapt_error,adapt_stdevs,adapt_precisions_all,adapt_precisions, \
        adapt_stdevs_q1s, adapt_stdevs_q3s,adapt_precisions_q1s, \
            adapt_precisions_q3s =[[],[]], [[],[]], [[],[]], [[],[]], [[],[]], \
                [[],[]], [[],[]], [[],[]]
                
    for p in range(len(parameters)):
        adapt_error[p] = np.median(adapt_errors[p])
        adapt_stdevs[p] = [np.median([s[i][p] for m,s,t in adapt_runs]) \
                       for i in range(steps+1)]
        adapt_stdevs_q1s[p] = [np.percentile([s[i][p] 
                                              for m,s,t in adapt_runs], 25) 
                               for i in range(steps+1)]
        adapt_stdevs_q3s[p] = [np.percentile([s[i][p] 
                                              for m,s,t in adapt_runs], 75) 
                               for i in range(steps+1)]

    adapt_precisions_all = [([(s[i][0]**2+(100*s[i][1])**2)*t[i] 
                                 for i in range(steps+1)]) 
                               for m,s,t in adapt_runs]
    adapt_precisions = [np.median([adapt_precisions_all[i][j] \
                      for i in range(runs)]) for j in range(steps+1)]
    adapt_precisions_q1s = [np.percentile([adapt_precisions_all[i][j] 
                                                  for i in range(runs)], 25) 
                                   for j in range(steps+1)]
    adapt_precisions_q3s = [np.percentile([adapt_precisions_all[i][j] 
                                                  for i in range(runs)], 75) 
                                   for j in range(steps+1)]
        
    print("Results for f and alpha:")
    print("Adaptive:\n- Standard deviation: %.2f; %.4f\n- Error: %.2f; %.4f"
          "\n- Final precision: %.2f" %  (adapt_stdevs[0][steps], 
                                               adapt_stdevs[1][steps],
                                               adapt_error[0], 
                                               adapt_error[1], 
                                               adapt_precisions[steps]))
    
    off_times = [np.median([t[i] for m,s,t in off_runs]) \
                       for i in range(steps+1)]
    off_error,off_stdevs,off_precisions_all,off_precisions, \
    off_stdevs_q1s, off_stdevs_q3s,off_precisions_q1s, \
        off_precisions_q3s =[[],[]], [[],[]], [[],[]], [[],[]], [[],[]], \
            [[],[]], [[],[]], [[],[]]
    
    for p in range(len(parameters)):
        off_error[p] = np.median(off_errors)
        off_stdevs[p] = [np.median([s[i][p] for m,s,t in off_runs]) \
                       for i in range(steps+1)]
        off_stdevs_q1s[p] = [np.percentile([s[i][p] 
                                            for m,s,t in off_runs], 25) 
                             for i in range(steps+1)]
        off_stdevs_q3s[p] = [np.percentile([s[i][p] for m,s,t in off_runs], 75) 
                             for i in range(steps+1)]

    off_precisions_all = [([(s[i][0]**2+(100*s[i][1])**2)*t[i] 
                                 for i in range(steps+1)]) 
                               for m,s,t in off_runs]
    off_precisions = [np.median([off_precisions_all[i][j] \
                      for i in range(runs)]) for j in range(steps+1)]
    off_precisions_q1s = [np.percentile([off_precisions_all[i][j] 
                                                  for i in range(runs)], 25) 
                                   for j in range(steps+1)]
    off_precisions_q3s = [np.percentile([off_precisions_all[i][j] 
                                                  for i in range(runs)], 75) 
                                   for j in range(steps+1)]
    
    print("Offline:\n- Standard deviation: %.2f; %.4f\n- Error: %.2f; %.4f"
          "\n- Final precision: %.2f" % (off_stdevs[0][steps], 
                                       off_stdevs[1][steps], 
                                       off_error[0],
                                       off_error[1],
                                       off_precisions[steps]))
    print("(n=%d (f_particles = %d); N=%d; f_max=%d; alpha_max=%.2f; "
          "runs=%d; 2d)" 
          % (N_particles,f_particles,steps,f_max, alpha_max,runs))
    
    fig, axs = plt.subplots(2,figsize=(8,8))
    
    p=0
    x1 = np.array([i for i in range(1,steps+1)])
    y1 = np.array([adapt_stdevs[p][i] for i in range(1,steps+1)])
    oy1 = np.array([off_stdevs[p][i] for i in range(1,steps+1)])
    axs[0].set_ylabel(r'$\sigma$')
    axs[0].plot(x1, y1, label='adaptive')
    axs[0].plot(x1, oy1, color='red', label='offline')
    q11 = np.array([adapt_stdevs_q1s[p][i] for i in range(1,steps+1)])
    q31 = np.array([adapt_stdevs_q3s[p][i] for i in range(1,steps+1)])
    oq11 = np.array([off_stdevs_q1s[p][i] for i in range(1,steps+1)])
    oq31 = np.array([off_stdevs_q3s[p][i] for i in range(1,steps+1)])
    axs[0].fill_between(x1,q11,q31,alpha=0.1)
    axs[0].fill_between(x1,oq11,oq31,alpha=0.1,color='red')
    axs[0].set_title('Frequency Estimation')
    axs[0].set_xlabel('Iteration number')
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    p=1
    y12 = np.array([adapt_stdevs[p][i] for i in range(1,steps+1)])
    oy1 = np.array([off_stdevs[p][i] for i in range(1,steps+1)])
    axs[1].set_ylabel(r'$\sigma$')
    axs[1].plot(x1, y12, label='adaptive')
    axs[1].plot(x1, oy1, color='red', label='offline')
    q11 = np.array([adapt_stdevs_q1s[p][i] for i in range(1,steps+1)])
    q31 = np.array([adapt_stdevs_q3s[p][i] for i in range(1,steps+1)])
    oq11 = np.array([off_stdevs_q1s[p][i] for i in range(1,steps+1)])
    oq31 = np.array([off_stdevs_q3s[p][i] for i in range(1,steps+1)])
    axs[1].fill_between(x1,q11,q31,alpha=0.1)
    axs[1].fill_between(x1,oq11,oq31,alpha=0.1,color='red')
    axs[1].set_title('Coherence Factor Estimation')
    axs[1].set_xlabel('Iteration number')
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    '''
    p=0
    x2 = np.array([adapt_times[i] for i in range(1,steps+1)])
    y2 = np.array([adapt_precisions[i] for i in range(1,steps+1)])
    ox2 = np.array([off_times[i] for i in range(1,steps+1)])
    oy2 = np.array([off_precisions[i] for i in range(1,steps+1)])
    q12 = np.array([adapt_precisions_q1s[i] for i in range(1,steps+1)])
    q32 = np.array([adapt_precisions_q3s[i] for i in range(1,steps+1)])
    oq12 = np.array([off_precisions_q1s[i] for i in range(1,steps+1)])
    oq32 = np.array([off_precisions_q3s[i] for i in range(1,steps+1)])
    axs[2].set_ylabel(r'$\sigma^2 \cdot \Delta t$')
    axs[2].loglog(x2, y2)
    axs[2].loglog(ox2, oy2,color='red')
    axs[2].fill_between(x2,q12,q32,alpha=0.1)
    axs[2].fill_between(ox2,oq12,oq32,alpha=0.1,color='red')
    axs[2].set_xlabel('Total accumulation time')
    axs[2].set_title('Precision')
    '''

    fig.subplots_adjust(hspace=0.55)
    
main()