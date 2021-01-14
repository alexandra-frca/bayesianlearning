# -*- coding: utf-8 -*-
"""
Hamiltonian learning implementation for a simple precession example, using both
offline and adaptive Bayesian inference.

The qubit is assumed to be initialized at state |+> for each iteration, and to
evolve under H = f*sigma_z/2, where f is the parameter to be estimated.

A sequential Monte Carlo approximation is used to represent the probability 
distributions, along with Liu-West resampling.

The evolution of the standard deviations the "precisions" (the variance times 
the cumulative evolution time) with the steps are plotted, and the final values 
of these quantities (in addition to the actual error) are printed.

The algorithm is repeated for a number of runs with randomly picked real
values, and medians are taken over all of them to get the results and graphs.
"""

import sys, random, numpy as np, matplotlib.pyplot as plt

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

def simulate(test_f, t):
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
    p: float
        The estimated probability of finding the particle at state |+>.
    '''
    p=np.cos(test_f*t/2)**2
    return p 

def resample(f_particle, distribution, resampled_distribution, 
             a=0.98, left_constraint = 0, right_constraint=10):
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
    left_constraint: float
        The leftmost bounds to be enforced for the particle's motion.
    right_constraint: float
        The rightmost bounds to be enforced for the particle's motion.
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

def adaptive_estimation(distribution, steps, k=1.25, guesses=1, precision=0):
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
    precision: float,optional
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
    steps = 100
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
        
    print("(n=%d; N=%d; f_max=%d; alpha=%.2f; runs=%d; 1d, SMC)" 
          % (N_particles,steps,f_max,alpha_real,runs))
        
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
