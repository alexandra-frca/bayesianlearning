# -*- coding: utf-8 -*-
"""
Hamiltonian learning implementation for a simple precession example, using both
offline and adaptive Bayesian inference.

The qubit is assumed to be initialized at state |+> for each iteration, and to
evolve under H = f*sigma_z/2, apart from the exponential decay resulting from 
the presence of decoherence. Estimation is performed for both the precession
frequency and the coherence time (or its inverse).

The evolution of the standard deviations the "precisions" (the variance times 
the cumulative evolution time) with the steps are plotted, and the final values 
of these quantities (in addition to the actual error) are printed.

The algorithm is repeated for a number of runs with randomly picked real
values, and medians are taken over all of them to get the results and graphs.
"""

import sys, random, matplotlib.pyplot as plt
import numpy as np
np.seterr(all='warn')
dim=2

N_particles = 2500 # Number of samples used to represent the probability
#distribution, using a sequential Monte Carlo approximation.

f_real, alpha_real = 0, 0 # The parameters we mean to estimate (a precession
#frequency, and a decay factor - the inverse of the coherence time) (will be 
#picked at random for each run).

def measure(t, particle=np.array([f_real,alpha_real]), tries=1):
    '''
    Simulates the measurement of the quantum system of the x component of spin 
    at a given time t after initialization at state |+>.
    
    Parameters
    ----------
    t: float
        The evolution time between the initialization and the projection.
    particle: [float], optional
        The set of dynamical parameters governing the system's evolution 
    (Default is [f_real,alpha_real]).
    tries: int, optional
        The amount of times the measurement is repeated (Default is 1).
        
    Returns
    -------
    1 if the result is |+>, 0 if it is |->.
    '''
    r = random.random()
    p = np.random.binomial(tries, 
                           p=(np.cos(f_real*t/2)**2*np.exp(-alpha_real*t)+
                              (1-np.exp(-alpha_real*t))/2))/tries
    if (r<p):
        return 1
    return 0

def simulate(particle, t):
    '''
    Provides an estimate for the likelihood  P(D=1|test_f,t) of an x-spin 
    measurement at time t yielding result |+>, given a set of parameters for 
    the fixed form Hamiltonian. 
    
    Parameters
    ----------
    particle: [float]
        The set of dynamical parameters to be used in the simulation.
    t: float
        The evolution time between the initialization and the projection.
        
    Returns
    -------
    p: float
        The estimated probability of finding the particle at state |+>.
    '''
    test_f, test_alpha = particle
    p=np.cos(test_f*t/2)**2*np.exp(-test_alpha*t)+(1-np.exp(-test_alpha*t))/2
    return p 

def resample(particle, distribution, resampled_distribution, 
             a=0.98, left_constraints = [0,0], right_constraints=[10,0.1]):
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
        The Liu-West filtering parameter (Default is 0.98).
    left_constraints: [float]
        The leftmost bounds to be enforced for the particle's motion.
    right_constraints: [float]
        The rightmost bounds to be enforced for the particle's motion.
    '''
    global dim
    
    current_means, current_stdevs = SMCparameters(distribution)
    current_f_mean, current_alpha_mean = current_means
    current_f_stdev, current_alpha_stdev = current_stdevs
    
    # Start with any invalid point.
    new_particle = np.array([left_constraints[i]-1 for i in range(dim)]) 
    
    # Get a proposal that satisfies the constraints.
    while any([new_particle[i]<=left_constraints[i] 
                   for i in range(len(new_particle))] + 
                  [new_particle[i]>=right_constraints[i] 
                   for i in range(len(new_particle))]):
        new_particle = np.array([np.random.normal(a*particle[i]+
                                                  (1-a)*current_means[i],
                                                  scale=(1-a**2)**0.5
                                                  *current_stdevs[i])
                                 for i in range(dim)])
        # Avoid repeated particles.
        if new_particle.tobytes() in resampled_distribution:
            new_particle = np.array([left_constraints[i]-1 for i in range(dim)]) 
        
    key = new_particle.tobytes()
    
    # Attribute uniform weights to the resampled particles.
    resampled_distribution[key] = 1/N_particles

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
    for key in distribution:
        particle = np.frombuffer(key,dtype='float64')
        p1 = simulate(particle,t)
        if (outcome==1):
            w = p1*distribution[key]
        if (outcome==0):
            w = (1-p1)*distribution[key]
        distribution[key] = w
        acc_weight += w
    
    # Normalize the weights.
    for key in distribution:
        w = distribution[key]/acc_weight
        distribution[key] = w
        acc_squared_weight += w**2 # The inverse participation ratio will be
        #used to decide whether to resample or not.
       
    # Check whether the resampling condition applies, and resample if so.
    if (1/acc_squared_weight <= threshold):
        resampled_distribution = {}
        for i in range(N_particles):
            key = random.choices(list(distribution.keys()), 
                                  weights=distribution.values())[0]
            particle = np.frombuffer(key,dtype='float64')
            resample(particle,distribution,
                     resampled_distribution)
        distribution = resampled_distribution
        
    return distribution

def SMCparameters(distribution, stdev=True):
    '''
    Calculates the mean and (optionally) standard deviation of a given 
    distribution.
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle=[f,alpha]:=[frequency,decay factor] (as a bit string)
        The distribution (SMC approximation) whose parameters are to be 
        calculated. This can also be a list if the weights are uniform.
    stdev: bool
        To be set to False if the standard deviation is not to be returned 
        (Default is True).
        
    Returns
    -------
    means: [float]
        The means of the distribution along its two dimensions: frequency and
        decay factor (the inverse of the coherence time), by this order.
    stdevs: [float]
        The standard deviation of the distribution along its two dimensions: 
        frequency and decay factor (the inverse of the coherence time), by this 
        order.
    '''
    global dim, N_particles
    means, meansquares = np.zeros(dim)
    for key in distribution:
        particle = np.frombuffer(key,dtype='float64')
        w = distribution[key]
        means += particle*w
        meansquares += particle**2*w
    
    if not stdev:
        return means
    
    stdevs = abs(means**2-meansquares)**0.5    
    return means,stdevs

first_offline_estimation = True
def offline_estimation(distribution, f_max, steps, increment=0.08):
    '''
    Estimates the precession frequency by defining a set of experiments, 
    performing them, and updating a given prior distribution according to their 
    outcomes (using Bayesian inference).
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle=[f,alpha]:=[frequency,decay factor] (as a bit string)
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
    global first_offline_estimation
    if first_offline_estimation is True:
        print("Offline estimation: increment=%.2f" % increment)
        first_offline_estimation = False
        
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
        distribution = bayes_update(distribution,t,m) 
        
        current_mean, current_stdev = SMCparameters(distribution)
        means.append(current_mean)
        stdevs.append(current_stdev) 
        
    cumulative_times = np.cumsum([i/(2*f_max) for i in range(steps+1)])
    return means, stdevs, cumulative_times

def expected_utility(distribution, t, scale):
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
        , and particle=[f,alpha]:=[frequency,decay factor] (as a bit string)
        The prior distribution (SMC approximation).
    t: float
        The measurement time for which the utility is to be computed.
    scale: [float]
        A list of factors defining the relative scale between the parameters,
        by the same order they are listed in the arrays representing particles.
        
    Returns
    -------
    utility: float
        The expectation value for the utility of the provided measurement time. 
    '''
    global dim
    # Obtain the probability of each oucome, given the current distribution.
    p1=0
    for key in distribution:
        particle = np.frombuffer(key,dtype='float64')
        p1 += simulate(particle,t)*distribution[key]
    p0 = 1-p1
        
    dist_0 = distribution.copy()
    dist_1 = distribution.copy()
    
    # Update the distribution assuming each of the possible outcomes.
    dist_0 = bayes_update(dist_0,t,0) 
    dist_1 = bayes_update(dist_1,t,1)
    
    # Compute the expected utility for each oucome.
    stdevs_0 = SMCparameters(dist_0)[1]
    stdevs_1 = SMCparameters(dist_1)[1]
    util_0 = -np.sum([(scale[i]*stdevs_0[i])**2 for i in range(dim)])
    util_1 = -np.sum([(scale[i]*stdevs_1[i])**2 for i in range(dim)])

    # Calculate the expected utility over all (both) outcomes.
    utility = p0*util_0 + p1*util_1
    
    return(utility)

def adaptive_guess(distribution, k, scale, guesses):
    '''
    Provides a guess for the evolution time to be used for a measurement,
    picked using the PGH, a particle guess heuristic (where the times are 
    chosen to be inverse to the distance between two particles sampled at 
    random from the distribution).
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle=[f,alpha]:=[frequency,decay factor] (as a bit string)
        The prior distribution (SMC approximation).
    k: float
        The proportionality constant to be used for the particle guess 
        heuristic.    
    scale: [float], optional
        A list of factors defining the relative scale between the parameters,
        by the same order they are listed in the arrays representing particles.
    guesses: int
        The amount of hypothesis to be picked for the time using the PGH; only 
        the one which maximizes the expected utility among this set will be  
        chosen.
        
    Returns
    -------
    adaptive_t: float
        The measurement time to be used.
    '''
    global dim
    adaptive_ts, utilities = [], []
    for i in range(guesses):
        delta=0
        while (delta==0):
            k1, k2 = random.choices(list(distribution.keys()), 
                                      weights=distribution.values(), k=2)
            p1, p2 = np.frombuffer(k1,dtype='float64'),\
                np.frombuffer(k2,dtype='float64')
            delta = np.sum([(scale[i]*(p1[i]-p2[i]))**2 for i in range(dim)])
        time = k/delta
        if (guesses==1):
            return(time)
        adaptive_ts.append(time)
    for t in adaptive_ts:
        utilities.append(expected_utility(distribution, t, scale))
        
    return(adaptive_ts[np.argmax(utilities)])
    
first_adaptive_estimation = True
def adaptive_estimation(distribution, steps, scale=[1.,100.], k=1.25,
                        guesses=1, precision=0):
    '''
    Estimates the precession frequency by adaptively performing a set of 
    experiments, using the outcome of each to update the prior distribution 
    (using Bayesian inference) as well as to decide the next experiment to be 
    performed.
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle=[f,alpha]:=[frequency,decay factor] (as a bit string)
        The prior distribution (SMC approximation).
    steps: int
        The maximum number of experiments to be performed.
    scale: [float], optional
        A list of factors defining the relative scale between the parameters,
        by the same order they are listed in the arrays representing particles
        (Default is [1.,100.]).
    guesses: int, optional
        The amount of hypothesis to be picked for the time; only the one which      
        maximizes the expected utility among this set will be chosen (Default 
        is 1).
        If this quantity is greater than one, the times will be chosen to be 
        inversely proportional to the distance between two particles picked at
        random from the current distribution (instead of to its standard 
        deviation), in order to introduce variability.
    k: float, optional
        The proportionality constant to be used for the particle guess 
        heuristic (Default is 1.25).
        
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
    
    global dim
    current_means, current_stdevs = SMCparameters(distribution)
    means, stdevs, cumulative_times = [], [], []
    means.append(current_means)
    stdevs.append(current_stdevs) 
    cumulative_times.append(0)
    
    if (guesses==1):
        adaptive_t = k/np.sum([scale[i]*(current_stdevs[i])**2 \
                                for i in range(dim)])
    else:
        adaptive_t = adaptive_guess(distribution,k,scale,guesses)
        
    cumulative_times.append(adaptive_t)

    for i in range(1,steps+1):
        m = measure(adaptive_t)
        # Update the distribution: get the posterior of the current iteration, 
        #which is the prior for the next.
        distribution = bayes_update(distribution,adaptive_t,m)
        
        current_means, current_stdevs = SMCparameters(distribution)
        means.append(current_means)
        stdevs.append(current_stdevs) 
        
        if any([current_stdevs[i]<precision*scale[i] 
                for i in range(dim)]): 
            for j in range(i+1,steps+1): # Fill in the rest to get fixed length
        #lists for the plots.
                means.append(current_means)
                stdevs.append(current_stdevs)
                cumulative_times.append(cumulative_times[i-1])
            break
            
        if (guesses==1):
            adaptive_t = k/np.sum([scale[i]*(current_stdevs[i])**2 \
                                    for i in range(dim)])
        else:
            adaptive_t = adaptive_guess(distribution,k,scale,guesses)
        
        cumulative_times.append(adaptive_t+cumulative_times[i-1])      
    return means, stdevs, cumulative_times

def main():
    global f_real, alpha_real, N_particles
    f_max, alpha_max = 10, 0.1
    f_particles = N_particles**0.5
    alpha_particles = N_particles/f_particles
    
    # Start with a uniform prior.
    fs = np.arange(f_max/N_particles,f_max+f_max/N_particles,
                   f_max/N_particles) 
    alphas = np.arange(0,alpha_max+alpha_max/alpha_particles,
                   alpha_max/(alpha_particles-1))
    
    prior = {}
    for f in fs:
        for alpha in alphas:
            particle = np.array([f,alpha])
            # Use bit strings as keys because numpy arrays are not hashable.
            key = particle.tobytes()
            prior[key] = 1/N_particles
    
    runs=10
    steps = 50
    adapt_runs, off_runs = [], []
    parameters = []
    
    for i in range(runs):
        f_real, alpha_real = f_max*random.random(), alpha_max*random.random()
        parameters.append(np.array([f_real,alpha_real]))
        
        adapt_runs.append(adaptive_estimation(prior.copy(),steps,precision=0))
        off_runs.append(offline_estimation(prior.copy(),f_max,steps))
        
        if (i%(runs/10)==0):
            sys.stdout.write('.');sys.stdout.flush();

    '''
    The indexes in adapt_runs/off_runs are, by order: 
        - Run number;
        - Desired quantity (0 for mean, 1 for stdev, 2 for cumulative_time)
        - Step number
        - Desired parameter (0 for frequency, 1 for alpha)
    '''
    adapt_errors = [[abs(adapt_runs[i][0][steps-1][p]-parameters[i][p]) 
                     for i in range(runs)] for p in range(2)]
    off_errors = [[abs(off_runs[i][0][steps-1][p]-parameters[i][p]) 
                   for i in range(runs)] for p in range(2)]  

    adapt_error,adapt_stdevs,adapt_precisions_all,adapt_precisions, \
        adapt_stdevs_q1s, adapt_stdevs_q3s = [[],[]], [[],[]], [[],[]],\
                [[],[]], [[],[]], [[],[]]
    
    for p in range(dim):
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

    print("\nResults for f and alpha:")
    print("Adaptive:\n- Standard deviation: %.2f; %.4f\n- Error: %.2f; %.4f"
          "\n- Final precision: %.2f" %  (adapt_stdevs[0][steps], 
                                               adapt_stdevs[1][steps],
                                               adapt_error[0], 
                                               adapt_error[1], 
                                               adapt_precisions[steps]))
    

    off_error,off_stdevs,off_precisions_all,off_precisions, off_stdevs_q1s,\
        off_stdevs_q3s = [[],[]], [[],[]], [[],[]], [[],[]], [[],[]], [[],[]]
    
    for p in range(dim):
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
    fig.subplots_adjust(hspace=0.55)
    
    p=0
    x1 = np.array([i for i in range(steps+1)])
    y1 = np.array([adapt_stdevs[p][i] for i in range(steps+1)])
    oy1 = np.array([off_stdevs[p][i] for i in range(steps+1)])
    axs[0].set_ylabel(r'$\sigma$')
    axs[0].plot(x1, y1, label='adaptive')
    axs[0].plot(x1, oy1, color='red', label='offline')
    q11 = np.array([adapt_stdevs_q1s[p][i] for i in range(steps+1)])
    q31 = np.array([adapt_stdevs_q3s[p][i] for i in range(steps+1)])
    oq11 = np.array([off_stdevs_q1s[p][i] for i in range(steps+1)])
    oq31 = np.array([off_stdevs_q3s[p][i] for i in range(steps+1)])
    axs[0].fill_between(x1,q11,q31,alpha=0.1)
    axs[0].fill_between(x1,oq11,oq31,alpha=0.1,color='red')
    axs[0].set_title('Frequency Estimation')
    axs[0].set_xlabel('Iteration number')
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    p=1
    y12 = np.array([adapt_stdevs[p][i] for i in range(steps+1)])
    oy1 = np.array([off_stdevs[p][i] for i in range(steps+1)])
    axs[1].set_ylabel(r'$\sigma$')
    axs[1].plot(x1, y12, label='adaptive')
    axs[1].plot(x1, oy1, color='red', label='offline')
    q11 = np.array([adapt_stdevs_q1s[p][i] for i in range(steps+1)])
    q31 = np.array([adapt_stdevs_q3s[p][i] for i in range(steps+1)])
    oq11 = np.array([off_stdevs_q1s[p][i] for i in range(steps+1)])
    oq31 = np.array([off_stdevs_q3s[p][i] for i in range(steps+1)])
    axs[1].fill_between(x1,q11,q31,alpha=0.1)
    axs[1].fill_between(x1,oq11,oq31,alpha=0.1,color='red')
    axs[1].set_title('Coherence Factor Estimation')
    axs[1].set_xlabel('Iteration number')
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    '''
    For plotting the "precision" (the product of the variance and the 
    cumulative time):
        
    adapt_times = [np.median([t[i] for m,s,t in adapt_runs]) \
         for i in range(steps+1)]
    adapt_precisions_q1s, adapt_precisions_q3s =[[],[]], [[],[]]
    
    adapt_precisions_q1s = [np.percentile([adapt_precisions_all[i][j] 
                                                  for i in range(runs)], 25) 
                                   for j in range(steps+1)]
    adapt_precisions_q3s = [np.percentile([adapt_precisions_all[i][j] 
                                                  for i in range(runs)], 75) 
                                   for j in range(steps+1)]
    
    off_times = [np.median([t[i] for m,s,t in off_runs]) \
                       for i in range(steps+1)]
    off_precisions_q1s, off_precisions_q3s =[[],[]], [[],[]]
    off_precisions_q1s = [np.percentile([off_precisions_all[i][j] 
                                                  for i in range(runs)], 25) 
                                   for j in range(steps+1)]
    off_precisions_q3s = [np.percentile([off_precisions_all[i][j] 
                                                  for i in range(runs)], 75) 
                                   for j in range(steps+1)]
    
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
    
main()
