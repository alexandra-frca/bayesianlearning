# -*- coding: utf-8 -*-
"""
Hamiltonian learning implementation for a simple precession example, using both
offline and adaptive Bayesian inference.
The qubit is assumed to be initialized at state |+> for each measurement, and 
to evolve under H = f*sigma_z/2, where f is the parameter to be estimated.
Markov Chain Monte Carlo is used to sample from the posterior distribution,
with Hamiltonian Monte Carlo and Metropolis-Hastings steps.
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

f_real = 0.5 # The actual precession frequency we mean to estimate.

alpha_real = 0 # The decay factor (the inverse of the coherence time).

f_min, f_max = 0,10 # The limits for the prior.

global iteration_n


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
    r = np.random.binomial(tries, 
                           p=(np.cos(f_real*t/2)**2*np.exp(-alpha*t)+
                              (1-np.exp(-alpha*t))/2))/tries
    return r

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
    Provides an estimate for the likelihood  P(D|test_f,t) of an x-spin 
    measurement at time t yielding a given result, given a test parameter for  
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
    if np.size(data)==2:
        t,outcome = data
        p = simulate_1(test_f,t)*(outcome==1)+\
            (1-simulate_1(test_f,t))*(outcome==0) 
    else:
        p = np.product([likelihood(datum, test_f) for datum in data])
    return p 

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
        p1 = simulate_1(particle,t)
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
        print("> Resampling at iteration %d." % iteration_n)
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
        
    mean,_ = SMCparameters(distribution)
    return(distribution,mean)

def offline_estimation(distribution, f_max, measurements, increment=0.08):
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
    global iteration_n
    ts = np.arange(1, measurements+1)*increment
    #ts = [0.1*(9/8)**k for k in range(measurements)]
    data = [(t,measure(t)) for t in ts]
    point, _ = SMCparameters(distribution)
    trajectory = [point]
    #print(point)
    for i in range(measurements):
        iteration_n = i
        t, outcome = data[i]
        distribution, point = bayes_update(distribution, t, outcome)
        trajectory.append(point)
        #print(point,proposal)
    plot_likelihood(data,points=trajectory)
    return point

def plot_likelihood(data, points=None, point_types=None,plot_gradient=False,every=3):
    '''
    Plots - on the interval [0,f_max[ - the likelihood function corresponding to 
    the given data (which is the product of the individual likelihoods of each 
    datum), as well as (optionally) the gradient of the log-likelihood and/or 
    (also optionally) a set of points (as an overposed scatter plot).
    If a list of labels indicating the methods used for the Markov transitions 
    used to get each point is given, the points will be colored according to 
    these labels.
    
    Parameters
    ----------
    data: [(int,float,int)]
        A vector of experimental results and controls, each datum being of the 
        form (M,theta,outcome), where 'M' and 'theta' are the controls used for 
        each experiment and 'outcome' is its result.
    points: [float], optional
        A list of consecutive x-coordinates to be plotted. The y-coordinates
        will be obtained by enumeration, so that the upward direction of the 
        y-axis denotes evolution (Default is None).
    point_types: [str], optional
        A list giving the methods used for the Markov transitions, by the same 
        order as in the list of points. These methods can be either "HMC" or 
        "MH", and will be used to color and label the points (Default is None).
    plot_gradient: bool, optional
        Whether to plot the gradient (Default is False).
    '''
    fig, axs = plt.subplots(1,figsize=(15,10))
    #axs.set_title("MCMC using HMC and MH proposals",pad=20,fontsize=18)
    axs.set_ylabel("Likelihood",fontsize=20)
    axs.set_xlabel("Frequency",fontsize=20)
    axs.tick_params(axis='y', colors="blue")
    axs.yaxis.label.set_color('blue')
    
    global f_max
    xx = np.arange(0.1,f_max,0.1)
    yy = [likelihood(data,x) for x in xx]
    axs.plot(xx,yy,linewidth=1.75,alpha=0.5)
    if points is not None:
        axs2 = axs.twinx()
        axs2.set_ylabel("Iteration number",fontsize=20)
        
        if point_types is None:
          selected_points = points[::every] # plot every #every point only
          y = [every*i for i in range(len(selected_points))]
          axs2.scatter(selected_points, y, 
                       s=75, c="black")
        else:
          unique_types = ["Starting point","HMC","MH"]
          for unique_type in unique_types:
              xi = [points[i] for i in range(len(points)) 
                  if point_types[i]==unique_type]
              yi = [i for i in range(len(points)) if point_types[i]==unique_type]
              color = "red" if unique_type == "HMC" \
                  else "black" if unique_type == "MH" else "blue"
              marker_type = "x" if unique_type == "Starting point" else "."
              axs2.scatter(xi, yi, marker=marker_type,s=300, c=color,
                          label=unique_type)
          axs2.legend(loc='upper right',fontsize=20)
    if plot_gradient:
        fig2, axs2 = plt.subplots(1,figsize=(12,8))
        axs2.set_title("Gradient",fontsize=18)
        yy2 = [U_gradient(data,x) for x in xx]
        axs2.plot(xx,yy2)

def main():
    global f_real, alpha_real, N_particles, f_max, f_real

    fs = np.arange(f_max/N_particles,f_max+f_max/N_particles,
                   f_max/N_particles) 
    prior = {}
    for f in fs:
        prior[f] = 1/N_particles # We consider a flat prior up to f_max.

    measurements = 100
     
    offline_estimation(prior,f_max,measurements)
    print(">> particles=%d; steps/measurements=%d; f_max=%d; alpha=%.2f; 1d, SMC-LWF" 
          % (N_particles,measurements,f_max,alpha_real))
    
    global total_HMC, accepted_HMC, total_MH, accepted_MH
    if (total_HMC+total_MH)!=0:
        print("* Percentage of HMC steps:  %.1f%%." 
              % (100*total_HMC/(total_HMC+total_MH)))
    if (total_HMC != 0):
        print("* Hamiltonian Monte Carlo: %d%% mean particle acceptance rate." 
              % round(100*accepted_HMC/total_HMC))
    if (total_MH != 0):
        print("* Metropolis-Hastings:     %d%% mean particle acceptance rate." 
              % round(100*accepted_MH/total_MH))
    
main()