# -*- coding: utf-8 -*-
"""
Hamiltonian learning implementation for a simple precession example, using both
offline and adaptive Bayesian inference.

The qubit is assumed to be initialized at state |+> for each iteration, and to
evolve under H = f*sigma_z/2, apart from the exponential decay resulting from 
the presence of decoherence. Estimation is performed for both the precession
frequency and the coherence time (or its inverse).

A sequential Monte Carlo approximation is used to represent the probability 
distributions, using Hamiltonian Monte Carlo and Metropolis-Hastings mutation 
steps.

The evolution of the standard deviations with the steps is plotted, and the 
final  values of these quantities (in addition to the actual error) are 
printed.

The algorithm is repeated for a number of runs with randomly picked real
values, and medians are taken over all of them to get the results and graphs.
"""

import sys, random, matplotlib.pyplot as plt
from autograd import grad, numpy as np
np.seterr(all='warn')
dim=2
total_HMC, accepted_HMC = 0, 0
total_MH, accepted_MH = 0, 0

N_particles = 25 # Number of samples used to represent the probability
#distribution, using a sequential Monte Carlo approximation.

f_real, alpha_real = 0, 0 # The parameters we mean to estimate (a precession
#frequency, and a decay factor - the inverse of the coherence time) (will be 
#picked at random for each run).

left_boundaries = np.array([0,0])
right_boundaries = np.array([10,0.1])

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

def simulate_1(particle, t):
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

def likelihood(data,particle):
    '''
    Provides an estimate for the likelihood  P(D|test_f,t) of x-spin 
    measurements yielding the input vector of data, given test parameters for  
    the fixed form Hamiltonian. 
    
    Parameters
    ----------
    data: [(float,int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    particle: [float]
        The set of dynamical parameters to be used for the likelihood.s
        
    Returns
    -------
    p: float
        The estimated probability of obtaining the input outcome. 
    '''
    if np.size(data)==2: # Single datum case.
        t,outcome = data
        p = simulate_1(particle,t)*(outcome==1)+\
            (1-simulate_1(particle,t))*(outcome==0) 
    else:
        p = np.product([likelihood(datum, particle) for datum in data])
    return p 

def target_U(data,particle):
    '''
    Evaluates the target "energy" associated to the joint likelihood of some 
    vector  of data, given a set of parameters for the fixed form Hamiltonian. 
    
    Parameters
    ----------
    data: [(float,int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    particle: [float]
        The set of dynamical parameters to be used for the likelihood.
        
    Returns
    -------
    U: float
        The value of the "energy".
    '''
    test_f, test_alpha = particle
    U = -np.sum([np.log(likelihood(datum, particle)) for datum in data])
    return (U)

def U_gradient(data,particle,autograd=False):
    '''
    Evaluates the gradient of the target "energy" associated to the likelihood 
    at a time t seen as a probability density, given a set of parameters for         
    the fixed form Hamiltonian. 
    
    Parameters
    ----------
    data: [(float,int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    particle: [float]
        The set of dynamical parameters to be used for the likelihood.
    autograd: bool, optional
        Whether to use automatic differenciation (Default is False).
        
    Returns
    -------
    DU: [float]
        The gradient of the "energy".
    '''
    test_f, test_alpha = particle
    if autograd:
        DU_f = grad(target_U,1)
        DU = np.array(DU_f(data,particle))
    else:
        DU = np.array([0.,0.])

        for (t,outcome) in data:
            f = np.cos(test_f*t/2)**2*np.exp(-test_alpha*t)+\
                (1-np.exp(-test_alpha*t))/2
            if outcome==1:
                DU+=np.array([+np.exp(-test_alpha*t)*t*np.sin(test_f*t)/(2*f)
                             ,(t*np.exp(-test_alpha*t)*np.cos(test_f*t/2)**2-
                             t*np.exp(-test_alpha*t)/2)/f])
            if outcome==0:
                DU+=np.array([-np.exp(-test_alpha*t)*t*np.sin(test_f*t)/\
                              (2*(1-f))
                             ,-(t*np.exp(-test_alpha*t)*np.cos(test_f*t/2)**2-
                             t*np.exp(-test_alpha*t)/2)/(1-f)])
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
def metropolis_hastings_step(data, particle, S=np.identity(2),
                             factor=0.005,
                             left_constraints=left_boundaries, 
                             right_constraints=right_boundaries):
    '''
    Performs a Metropolis-Hastings mutation on a given particle, using a 
    gaussian function for the proposals.
    
    Parameters
    ----------
    data: [(float,int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    particle: [float]
        The particle to undergo a mutation step.
    S: [[float]]
        The covariance matrix that will determine the standard deviations
        of the normal distributions used for the proposal in each dimension
        (the dth diagonal element corresponding to dimension d).
    factor: float
        The factor M is to be multiplied by to get the actual standard 
        deviations.
    left_constraints: [float]
        The leftmost bounds to be enforced for the particle's motion.
    right_constraints: [float]
        The rightmost bounds to be enforced for the particle's motion.
        
    Returns
    -------
    particle: [float]
        The mutated particle.
    p: float
        The acceptance probability to be used for the evolved particle as a 
        Monte Carlo proposal.
    '''
    global first_metropolis_hastings_step
    if first_metropolis_hastings_step:
      if not np.array_equal(S,np.identity(2)):
          Cov = "Cov"
      else:
          Cov = "I"
      print("MH:  S=%s, factor=%.4f" % (Cov,factor))
      first_metropolis_hastings_step = False

    global dim
    Sigma = factor*S
    # Start with any invalid point.
    new_particle = np.array([left_constraints[i]-1 for i in range(dim)]) 
    
    # Get a proposal that satisfies the constraints.
    while any([new_particle[i]<left_constraints[i] for i in range(dim)] + 
                  [new_particle[i]>right_constraints[i] for i in range(dim)]):
        new_particle = np.array([np.random.normal(particle[i], Sigma[i][i])
                                 for i in range(dim)])

    # Compute the probabilities of transition for the acceptance probability.
    inverse_transition_prob = \
        np.product([gaussian(particle[i],new_particle[i],
                                                  Sigma[i][i]) 
                                          for i in range(dim)])
    transition_prob = np.product([gaussian(new_particle[i],particle[i],
                                       Sigma[i][i]) for i in range(dim)])

    p = likelihood(data,new_particle)*inverse_transition_prob/ \
        (likelihood(data,particle)*transition_prob)
    return new_particle,p
    
def simulate_dynamics(data, initial_momentum, initial_particle, M,L,eta,
                      left_constraints=left_boundaries, 
                      right_constraints=right_boundaries):  
    '''
    Simulates Hamiltonian dynamics for a given particle, using leapfrog 
    integration.
    
    Parameters
    ----------
    data: [(float,int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    initial_momentum: [float]
        The starting momentum vector. 
    initial_particle: [float]
        The particle for which the Hamiltonian dynamics is to be simulated.
    M: [[float]]
        The mass matrix/Euclidean metric to be used when simulating the
        Hamiltonian dynamics (a HMC tuning parameter).
    L: int
        The amount of integration steps to be used when simulating the 
        Hamiltonian dynamics (a HMC tuning parameter).
    eta: float
        The integration stepsize to be used when simulating the Hamiltonian 
        dynamics (a HMC tuning parameter).
    left_constraints: [float]
        The leftmost bounds to be enforced for the particle's motion.
    right_constraints: [float]
        The rightmost bounds to be enforced for the particle's motion.
        
    Returns
    -------
    particle: [float]
        The particle having undergone motion.
    p: float
        The acceptance probability to be used for the evolved particle as a 
        Monte Carlo proposal.
    '''
    global dim    
    M_inv = np.linalg.inv(M)
    new_particle = initial_particle
    DU = U_gradient(data,new_particle)
    
    # Perform leapfrog integration according to Hamilton's equations.
    new_momentum = np.add(initial_momentum,-0.5*eta*DU)
    for l in range(L):
        new_particle = np.add(new_particle,np.dot(M_inv,eta*new_momentum))
        
        # Enforce the constraints that both the frequency and the decay 
        #parameter lie within the prior distribution. 
        # Should a limit be crossed, the position and momentum are chosen such 
        #that the particle "rebounds".
        for i in range(dim):
            if (new_particle[i] < left_constraints[i]):
                new_particle[i] = left_constraints[i]+\
                    (left_constraints[i]-new_particle[i])
                new_momentum[i] = -new_momentum[i]
            if (new_particle[i] > right_constraints[i]):
                new_particle[i] = right_constraints[i]-\
                    (new_particle[i]-right_constraints[i])
                new_momentum[i] = -new_momentum[i]

        if (l != L-1):
            DU = U_gradient(data,new_particle)
            new_momentum = np.add(new_momentum,-eta*DU)     
    new_momentum = np.add(new_momentum,-0.5*eta*DU)
    
    # Compute the acceptance probability.
    p = np.exp(target_U(data,initial_particle)
               -target_U(data,new_particle)
               +np.sum(np.linalg.multi_dot(
                   [initial_momentum,M_inv,initial_momentum]))/2
               -np.sum(np.linalg.multi_dot(
                   [new_momentum,M_inv,new_momentum]))/2)
    '''
    if (p<0.1):
        sys.exit('p too small')
    '''
    return new_particle, p
        
first_hamiltonian_MC_step = True
def hamiltonian_MC_step(data, particle, 
                        M=np.identity(2), L=20, eta=0.001, threshold=0.1):
    '''
    Performs a Hamiltonian Monte Carlo mutation on a given particle.
    
    Parameters
    ----------
    data: [(float,int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    particle: [float]
        The particle to undergo a mutation step.
    M: [[float]], optional
        The mass matrix/Euclidean metric to be used when simulating the
        Hamiltonian dynamics (a HMC tuning parameter) (Default is the identity
        matrix).
    L: int, optional
        The amount of integration steps to be used when simulating the 
        Hamiltonian dynamics (a HMC tuning parameter) (Default is 20).
    eta: float, optional
        The integration stepsize to be used when simulating the Hamiltonian 
        dynamics (a HMC tuning parameter) (Default is exp(-3)).
    threshold: float, optional
        The highest HMC acceptance rate that should trigger a Metropolis-
        -Hastings mutation step (as an alternative to a  HMC mutation step) 
        (Default is 0.1). 
        
    Returns
    -------
    particle: [float]
        The mutated particle.
    '''
    if (threshold<1):
        # Perform a Hamiltonian Monte Carlo mutation.
        global first_hamiltonian_MC_step
        if first_hamiltonian_MC_step:
            if not np.array_equal(M,np.identity(2)):
                mass = "Cov^-1"
            else:
                mass = "I"
            print("HMC: M=%s, L=%d, eta=%.10f" % (mass,L,eta))
            first_hamiltonian_MC_step = False
            
        global total_HMC, accepted_HMC, total_MH, accepted_MH
        initial_momentum = np.random.multivariate_normal([0,0], M)
        new_particle, p = \
            simulate_dynamics(data,initial_momentum,particle,M,L,eta)
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
        Cov = np.linalg.inv(M)
        new_particle, p = metropolis_hastings_step(data,particle,S=Cov)
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
        return(new_particle)
    else:
        return(particle)

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
        , with (key,value):=(particle,importance weight) 
        , and particle=[f,alpha]:=[frequency,decay factor] (as a bit string)
        The prior distribution (SMC approximation). When returning, it will 
        have been updated according to the provided experimental datum.
    '''
    global N_particles
    
    # Perform a correction step by re-weighting the particles according to 
    #the last datum obtained.
    t,outcome = data[-1]
    for key in distribution:
        particle = np.frombuffer(key,dtype='float64')
        p1 = simulate_1(particle,t)
        if (outcome==1):
            w = p1
        if (outcome==0):
            w = 1-p1
        distribution[key] = w
    
    # Perform an importance sampling step (with replacement) according to the               
    #updated weights.
    selected_particles = random.choices(list(distribution.keys()), 
                                          weights=distribution.values(),
                                          k=N_particles)
    
    stdevs = SMCparameters(selected_particles)[1]
    Cov = np.diag(stdevs**2)
    
    # Check for singularity (the covariance matrix must be invertible).
    if (np.linalg.det(Cov) == 0): 
        return

    Cov_inv = np.linalg.inv(Cov)

    # Perform a mutation step on each selected particle, imposing that the
    #particles be unique.
    distribution.clear()
    for key in selected_particles:
        repeated = True
        while (repeated == True):
            particle = np.frombuffer(key,dtype='float64')
            mutated_particle = hamiltonian_MC_step(data,particle,
                                                   M=Cov_inv)
            key = mutated_particle.tobytes()
            if (key not in distribution):
                repeated = False
        distribution[key] = 1

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
        means += particle
        meansquares += particle**2
    means = means/N_particles
    meansquares = meansquares/N_particles
    
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
    data = []
    
    ts = np.arange(1, steps+1)*increment
    for t in ts:
        data.append((t,measure(t)))
        # Update the distribution: get the posterior of the current iteration, 
        #which is the prior for the next.
        bayes_update(data, distribution) 
        
        current_mean, current_stdev = SMCparameters(distribution)
        means.append(current_mean)
        stdevs.append(current_stdev) 
        
    cumulative_times = np.cumsum([i/(2*f_max) for i in range(steps+1)])
    return means, stdevs, cumulative_times

def expected_utility(data,distribution, t, scale):
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
        p1 += likelihood(particle,t)
    p0 = 1-p1
        
    dist_0 = distribution.copy()
    dist_1 = distribution.copy()
    
    # Update the distribution assuming each of the possible outcomes.
    bayes_update(data+[(t,0)],dist_0) 
    bayes_update(data+[(t,1)],dist_1)
    
    # Compute the expected utility for each oucome.
    stdevs_0 = SMCparameters(dist_0)[1]
    stdevs_1 = SMCparameters(dist_1)[1]
    util_0 = -np.sum([(scale[i]*stdevs_0[i])**2 for i in range(dim)])
    util_1 = -np.sum([(scale[i]*stdevs_1[i])**2 for i in range(dim)])

    # Calculate the expected utility over all (both) outcomes.
    utility = p0*util_0 + p1*util_1
    
    return(utility)


def adaptive_guess(data, distribution, k, scale, guesses):
    '''
    Provides a guess for the evolution time to be used for a measurement,
    picked using the PGH, a particle guess heuristic (where the times are 
    chosen to be inverse to the distance between two particles sampled at 
    random from the distribution).
    
    Parameters
    ----------
    data: [(float,int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
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
    guesses: int, optional
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
            delta = np.sum([(scale[i]*(p1[i]-p2[i]))**2 
                            for i in range(dim)])**0.5
        time = k/delta
        if (guesses==1):
            return(time)
        adaptive_ts.append(time)
    for t in adaptive_ts:
        utilities.append(expected_utility(data, distribution, t, scale))
        
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
        A list of the consecutive distribution means, including the prior's 
        and the ones resulting from every intermediate step.
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
    data = []
    
    if (guesses==1):
        adaptive_t = k/np.sum([scale[i]*(current_stdevs[i])**2 \
                                for i in range(dim)])
    else:
        adaptive_t = adaptive_guess(distribution,k,scale,guesses)
        
    cumulative_times.append(adaptive_t)

    for i in range(1,steps+1):
        data.append((adaptive_t,measure(adaptive_t)))
        # Update the distribution: get the posterior of the current iteration, 
        #which is the prior for the next.
        bayes_update(data, distribution)
        
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
            adaptive_t = adaptive_guess(data,distribution,k,scale,guesses)
      
        cumulative_times.append(adaptive_t+cumulative_times[i-1])      
    return means, stdevs, cumulative_times

def main():
    global f_real, alpha_real, N_particles, right_boundaries, dim
    f_max, alpha_max = right_boundaries
    f_particles = N_particles**0.5
    alpha_particles = N_particles/f_particles
    
    # Start with a uniform prior. 
    fs = np.arange(f_max/f_particles,f_max+f_max/f_particles,
                   f_max/f_particles) 
    alphas = np.arange(0,alpha_max+alpha_max/alpha_particles,
                   alpha_max/(alpha_particles-1))
    prior = {}
    for f in fs:
        for alpha in alphas:
            particle = np.array([f,alpha])
            # Use bit strings as keys because numpy arrays are not hashable.
            key = particle.tobytes()
            prior[key] = 1/N_particles
    
    runs=1
    steps = 10
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