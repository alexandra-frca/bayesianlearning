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

N_particles = 1 # Number of samples used to represent the probability
#distribution, using a sequential Monte Carlo approximation.

f_real = 0.5 # The actual precession frequency we mean to estimate.

alpha_real = 0 # The decay factor (the inverse of the coherence time).

f_min, f_max = 0,10 # The limits for the prior.



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

def loglikelihood(data, test_f):
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
        DU = np.sum([(outcome==1)*(t*np.sin(test_f*t/2)/np.cos(test_f*t/2))+
                     (outcome==0)*(-t*np.sin(test_f*t/2)*np.cos(test_f*t/2)/
                                   (1-np.cos(test_f*t/2)**2)) 
                     for (t,outcome) in usable_data])
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

#sigma = 0.1 works the best alone, though low acceptance rate (~5%) and not 
#very far, increment=0.08. Along with HMC use sth like 0.01-0.05
first_metropolis_hastings_step = True
def metropolis_hastings_step(data, particle, sigma=0.05,
                             left_constraint=f_min,right_constraint=f_max): 
    '''
    Performs a Metropolis-Hastings mutation on a given particle, using a 
    gaussian function for the proposals.
    
    Parameters
    ----------
    data: [(float,int)]
        A vector of experimental results and controls, each datum being of the 
        form (time,outcome), where 'time' is the control used for each 
        experiment and 'outcome' is its result.
    particle: float
        The particle to undergo a mutation step.
    sigma: float, optional
        The standard deviation to be used in the normal distribution to generate
        proposals (Default is 0.01).
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
      print("MH:  sigma=%s" % (sigma))
      first_metropolis_hastings_step = False

    # Start with any invalid value.
    new_particle = left_constraint-1
    
    # Get a proposal that satisfies the constraints.
    while (new_particle < left_constraint or new_particle > right_constraint):
        new_particle = np.random.normal(particle, sigma)
        #print("sugg",new_particle)
        
    # Compute the probabilities of transition for the acceptance probability.
    p = likelihood(data,new_particle)*gaussian(particle,new_particle,
                                                    sigma)/ \
        (likelihood(data,particle)*gaussian(new_particle,particle,sigma))
    return new_particle,p

def simulate_dynamics(data, initial_momentum, initial_particle, m, L, eta,
                      left_constraint = f_min, right_constraint=f_max):    
    '''
    Simulates Hamiltonian dynamics for a given particle, using leapfrog 
    integration.
    
    Parameters
    ----------
    data: [(float,int)]
        A vector of experimental results and controls, each datum being of the 
        form (time,outcome), where 'time' is the control used for each 
        experiment and 'outcome' is its result.
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
        while (new_particle < left_constraint or 
               new_particle > right_constraint):
            if (new_particle < left_constraint):
                new_particle = left_constraint+(left_constraint-new_particle)
                new_momentum = -new_momentum
            if (new_particle > right_constraint): 
                new_particle = right_constraint-(new_particle-right_constraint)
                new_momentum = -new_momentum
        if (l != L-1):
            DU = U_gradient(data,new_particle)
            new_particle = new_particle - eta*DU
    new_momentum = new_momentum - 0.5*eta*DU

    p = np.exp(-loglikelihood(data,initial_particle)-\
               (-loglikelihood(data,new_particle))+\
                   initial_momentum**2/(2*m)-new_momentum**2/(2*m))
    return new_particle, p
        
first = True

# works well for m=0.1, L=20,eta=10**-3, threshold=0.01, measurements=100,
#steps=100, increment=0.08
def hamiltonian_MC_step(data, point, m=0.1, L=20, eta=5*10**-3,
                        threshold=0.01):
    '''
    Performs a Hamiltonian Monte Carlo mutation on a given particle.
    
    Parameters
    ----------
    data: [(float,int)]
        A vector of experimental results and controls, each datum being of the 
        form (time,outcome), where 'time' is the control used for each 
        experiment and 'outcome' is its result.
    particle: float
        The frequency particle to undergo a transition.
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
    point: float
        The point after the suggested transition (i.e. a frequency parameter).
    proposal: str
        The mechanism used for the Markov transition ("HMC" or "MH").
    '''
    if (threshold<1):
        # Perform a Hamiltonian Monte Carlo mutation.
        global first
        if first:
            print("HMC: m=%s, L=%d, eta=%f" % (m,L,eta))
            first = False
            
        global total_HMC, accepted_HMC, total_MH, accepted_MH
        initial_momentum = np.random.normal(0, scale=m)
        new_point, p = simulate_dynamics(data,initial_momentum,point,m,L,eta)
    else:
        p = 0
        
    # If the Hamiltonian Monte Carlo acceptance probability is too low,
    #a Metropolis-Hastings mutation will be performed instead.
    # This is meant to saufegard the termination of the program if the leapfrog
    #integration is too inaccurate for a given set of parameters and experiment
    #controls (which tends to happen close to or at the assymptotes of the log-
    #-likelihood).
    
    if (p < threshold):
        proposal = "MH"
        MH = True
        new_point, p = metropolis_hastings_step(data,point)
        total_MH += 1
    else:
        proposal = "HMC"
        MH = False
        total_HMC += 1
        
    a = min(1,p) 
    if (np.random.rand() < a):
        if MH:
            accepted_MH += 1
        else:
            accepted_HMC += 1
        return(new_point, proposal)
    else:
        return(point, proposal)

def offline_estimation(point, f_max, measurements, steps, increment=0.08):
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
    ts = np.arange(1, measurements+1)*increment
    #ts = [0.1*(9/8)**k for k in range(measurements)]
    data = [(t,measure(t)) for t in ts]
    
    trajectory, proposals = [point],["Starting point"]
    #print(point)
    for i in range(steps):
        point, proposal = hamiltonian_MC_step(data,point)
        trajectory.append(point)
        proposals.append(proposal)
        #print(point,proposal)
    plot_likelihood(data,points=trajectory, point_types=proposals)
    return point

def plot_likelihood(data, points=None, point_types=None,plot_gradient=False):
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
    axs.set_title("MCMC using HMC and MH proposals",pad=20,fontsize=18)
    axs.set_ylabel("Likelihood",fontsize=14)
    axs.set_xlabel("Frequency",fontsize=14)
    axs.tick_params(axis='y', colors="blue")
    axs.yaxis.label.set_color('blue')
    
    global f_max
    xx = np.arange(0.1,f_max,0.1)
    yy = [likelihood(data,x) for x in xx]
    axs.plot(xx,yy,linewidth=1.75,alpha=0.5)
    if points is not None:
        axs2 = axs.twinx()
        axs2.set_ylabel("Step number",fontsize=14)
        
        unique_types = ["Starting point","HMC","MH"]
        for unique_type in unique_types:
            xi = [points[i] for i in range(len(points)) 
                 if point_types[i]==unique_type]
            yi = [i for i in range(len(points)) if point_types[i]==unique_type]
            color = "red" if unique_type == "HMC" \
                else "black" if unique_type == "MH" else "blue"
            marker_type = "x" if unique_type == "Starting point" else "."
            axs2.scatter(xi, yi, marker=marker_type,s=10, c=color,
                         label=unique_type)
        axs2.legend(loc='upper right',fontsize=14)
    if plot_gradient:
        fig2, axs2 = plt.subplots(1,figsize=(12,8))
        axs2.set_title("Gradient",fontsize=18)
        yy2 = [U_gradient(data,x) for x in xx]
        axs2.plot(xx,yy2)

def main():
    global f_real, alpha_real, N_particles, f_max, f_real
    start = f_max/2
    measurements = 100
    steps=100
     
    offline_estimation(start,f_max,measurements,steps)
    print("n=%d; N=%d; measurements=%d; f_max=%d; alpha=%.2f; 1d, SHMC" 
          % (N_particles,steps,measurements,f_max,alpha_real))
    
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