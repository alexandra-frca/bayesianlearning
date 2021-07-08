# -*- coding: utf-8 -*-
"""
Quantum parameter learning implementation for estimating T1, the energy loss 
characteristic time. Comments assume a frequency because this was copied from
the ramsey/precession examples but same thing applies
A sequential Monte Carlo approximation is used to represent the probability 
distributions, using Hamiltonian Monte Carlo and/or Metropolis-Hastings mutation 
steps.
The evolution of the standard deviations with the steps is plotted, and the 
final  values of these quantities (in addition to the actual error) are 
printed.
The algorithm is repeated for a number of runs with randomly picked real
values, and medians are taken over all of them to get the results and graphs.
"""

import sys, copy, random, pickle, matplotlib.pyplot as plt
from autograd import grad, numpy as np
import warnings
warnings.simplefilter("default") # error, ignore, default, always, once
np.seterr(all='warn')
np.seterr(under='ignore')
dim=1
total_HMC, accepted_HMC = 0, 0
total_MH, accepted_MH = 0, 0

N_particles = 50 # Number of samples used to represent the probability
#distribution, using a sequential Monte Carlo approximation.

T1_real = 0 

left_boundaries = np.array([10])
right_boundaries = np.array([100])

def measure(t):
    '''
    Simulates the measurement of the quantum system of the z component of spin 
    at a given time t.
    
    Parameters
    ----------
    t: float
        The evolution time between the initialization and the projection.
    particle: [float], optional
        The set of real dynamical parameters governing the system's evolution 
    (Default is [f_real,alpha_real]).
        
    Returns
    -------
    1 if the result is |1>, 0 if it is |0>.
    '''
    r = np.random.binomial(1, p=np.exp(-t/T1_real))
    return r

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
        The estimated probability of finding the particle at state |1>.
    '''
    T1 = particle[0]
    p = np.exp(-t/T1)
    return p 

def likelihood(data,particle):
    '''
    Provides an estimate for the likelihood  P(D|test_f,t) of z-spin 
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
        t,outcome = data if len(data)==2 else data[0] # May be wrapped in array.
        p = simulate_1(particle,t) if outcome==1 else (1-simulate_1(particle,t))
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
    if autograd:
        DU_f = grad(target_U,1)
        DU = np.array(DU_f(data,particle))
    else:
        T1s = particle[0]
        DU = np.array([0.])
        for (t,outcome) in data:
            L1 = 0.5*np.exp(-t/T1)+0.5
            DL1 = (t/T1**2)*0.5*np.exp(-t/T1)
            if outcome==1:
                DU += np.array([-DL1/L1])
            if outcome==0:
                L0 = 1-L1
                DL0 = -DL1
                DU += np.array([-DL0/L0])
    return(DU)
    
first_metropolis_hastings_step = True
def metropolis_hastings_step(data, particle, S=np.identity(2),
                             factor=0.2,
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
    global first_metropolis_hastings_step, dim
    if first_metropolis_hastings_step:
      if not np.array_equal(S,np.identity(dim)):
          Sigma = "Sigma"
      else:
          Sigma = "I"
      print("MH:  S=%s, factor=%.4f" % (Sigma,factor))
      first_metropolis_hastings_step = False

    Sigma = factor*S
    # Start with any invalid point.
    new_particle = np.array([left_constraints[i]-1 for i in range(dim)]) 
    
    # Get a proposal that satisfies the constraints.
    while any([new_particle[i]<left_constraints[i] for i in range(dim)] + 
                  [new_particle[i]>right_constraints[i] for i in range(dim)]):
        new_particle = np.array([np.random.normal(particle[i], Sigma[i][i])
                                 for i in range(dim)])

    p = 1 if likelihood(data,particle)==0 \
        else likelihood(data,new_particle)/likelihood(data,particle)
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
        DU = U_gradient(data,new_particle)
        if (l != L-1):
            new_momentum = np.add(new_momentum,-eta*DU)     
    new_momentum = np.add(new_momentum,-0.5*eta*DU)
    
    # Compute the acceptance probability.
    p = np.exp(target_U(data,initial_particle)
               -target_U(data,new_particle)
               +np.sum(np.linalg.multi_dot(
                   [initial_momentum,M_inv,initial_momentum]))/2
               -np.sum(np.linalg.multi_dot(
                   [new_momentum,M_inv,new_momentum]))/2)
    return new_particle, p
        
first_hamiltonian_MC_step = True
def hamiltonian_MC_step(data, particle, 
                        M=np.identity(2), L=20, eta=0.1, threshold=1):
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
        initial_momentum = np.random.multivariate_normal(np.zeros(dim), M)
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

resampler_calls = 0
first_bayes_update = True
def bayes_update(data, distribution, threshold=0.8):
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
    global first_bayes_update
    if first_bayes_update is True:
        print("Bayes update: resampling threshold = ", threshold)
        first_bayes_update = False

    global N_particles, resampler_calls
    acc_weight, acc_squared_weight = 0, 0
    
    # Perform a correction step by re-weighting the particles according to 
    #the last datum obtained.
    newest_datum = data[-1]
    for key in distribution:
        particle = np.frombuffer(key,dtype='float64')
        new_weight = likelihood(newest_datum,particle)*distribution[key]
        distribution[key] = new_weight
        acc_weight += new_weight

    # Normalize the weights.
    for key in distribution:
        w = distribution[key]/acc_weight
        distribution[key] = w
        acc_squared_weight += w**2 # The inverse participation ratio will be
        #used to decide whether to resample.

    resampled = False
    if (1/acc_squared_weight <= threshold*N_particles):
        resampled = True
        resampler_calls += 1
    
        # Perform an importance sampling step (with replacement) according to the               
        #updated weights.
        selected_particles = random.choices(list(distribution.keys()), 
                                              weights=distribution.values(),
                                              k=N_particles)
        
        stdevs = SMCparameters(selected_particles,list=True)[1]
        Cov = np.diag(stdevs**2)
        
        # Check for singularity (the covariance matrix must be invertible).
        if (np.linalg.det(Cov) == 0): 
            print("Non-invertible covariance matrix.")
            return

        Cov_inv = np.linalg.inv(Cov)

        # Perform a mutation step on each selected particle, imposing that the
        #particles be unique.
        distribution.clear()
        for key in selected_particles:
            particle = np.frombuffer(key,dtype='float64')
            particle = hamiltonian_MC_step(data,particle,M=Cov_inv)
            key = particle.tobytes()
            if (key not in distribution):
                distribution[key] = 1/N_particles
            else:
                distribution[key] += 1/N_particles

    return distribution,resampled

def SMCparameters(distribution, stdev=True, list=False):
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
    list: bool, optional
        To be set to True if the distribution is given as a list (as opposed to
        a dictionary) (weights are then assumed even) (Default is False).
        
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
    means, meansquares = np.zeros(dim),np.zeros(dim)
    for key in distribution:
        particle = np.frombuffer(key,dtype='float64')
        if list: # Distribution is given as a dictionary with implicitly 
        #uniform weights.
            w = 1/len(distribution)
        else: # Distribution is given by a dictionary with values as weights.
            w = distribution[key]
        means += particle*w
        meansquares += particle**2*w
    if not stdev:
        return means
    stdevs = abs(means**2-meansquares)**0.5
    return means,stdevs

def meansquarederror(distribution, real_parameters):
    '''
    Calculates the mean squared error given an SMC distribution.
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle=[f,alpha]:=[frequency,decay factor] (as a bit string)
        The distribution (SMC approximation) whose parameters are to be 
        calculated. This can also be a list if the weights are uniform.
    real_parameters: [float]
        The set of real parameters.
        
    Returns
    -------
    r: [float]
        The list of mean squared errors along each dimension (for each 
        parameter).
    '''
    global dim, N_particles
    r = np.zeros(dim)
    for key in distribution:
        particle = np.frombuffer(key,dtype='float64')
        w = distribution[key]
        r += (particle-real_parameters)**2*w
    return r

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
    e = np.exp(power)
    if not normalize:
        return e
    norm = (2*np.pi*sigma**2)**0.5  
    e = e/norm
    return e

def weighted_kernel_density_estimate(x, points, stdev):
    '''
    Computes the value of a (weighted) kernel density estimate at some point.
    
    Parameters
    ----------
    x: float 
        The point at which the estimate should be evaluated. 
    points: [([float],float)]
        A list of (point,weight) tuples, where `particle` is a 1-d coordinate 
        vector.
    stdev: float 
        The standard deviation of the points, to be used to choose the 
        bandwidth for the KDE.
        
    Returns
    -------
    kde: float
        The value of the kernel density estimate.
    '''
    nonzero = [p for p in points if p[1]>0]
    n = len(nonzero)
    h = 1.06*n**-0.2*stdev
    kde = 0
    for point in points:
        p,w = point
        kde += gaussian(x,p,h,normalize=True)/n
        # Division by h is already accounted for by the normalization.
    return(kde)

def dict_to_list(distribution):
    '''
    Converts a dictionary representation of a distribution to a list-of-tuples
    representation.
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The distribution to be converted (SMC approximation).
        
    Returns
    -------
    particles: [([float],float)]
        A list of (particle,weight) tuples, where `particle` is a parameter 
        vector.
    '''
    particles = []
    for key in distribution:
        particle = np.frombuffer(key,dtype='float64')
        weight = distribution[key]
        particles.append((particle,weight))
    return particles

def plot_curve(xs,ys,note=""):
    '''
    Plots a curve given a list of x-coordinates and the corresponding list of
    y-coordinates.
    
    Parameters
    ----------
    xs: [float]
        The first coordinates of the points to make up the curve.
    ys: [float]
        The second coordinates of the points to make up the curve, by the same
        order as in `xs`.
    '''
    fig, axs = plt.subplots(1,figsize=(8,8))
    axs.plot(xs,ys,color="black",linewidth=1)
    axs.set_title("Kernel density estimate %s" % note)
    axs.set_xlabel("Frequency")
    plt.xlim(left_boundaries,right_boundaries)

def plot_kde(distribution,note=""):
    '''
    Computes and plots the kernel density estimate of a distribution.
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The distribution of interest, to be represented by a KDE.
    '''
    stdev = SMCparameters(distribution)[1]
    ks =  dict_to_list(distribution)
    xs = np.arange(left_boundaries,right_boundaries,0.05)
    ys = [weighted_kernel_density_estimate(x,ks,stdev) for x in xs]
    plot_curve(xs,ys,note=note)

def show_results(off_runs,off_dicts,T1s,parameters):
    '''
    The indexes in off_runs are, by order: 
        - Run number;
        - Desired quantity (0 for mean, 1 for stdev, 2 for cumulative_time)
        - Step number
        - Desired parameter (0 for frequency, 1 for alpha)
    '''
    runs = len(off_runs)
    steps = len(off_runs[0][0])-1

    # Get run with the median last step variance to print its mean squared error
    #instead of calculating all of them since we won't plot them.
    stdevs = [s[steps-1][0] for m,s,t in off_runs]
    median = np.percentile(stdevs, 50, interpolation='nearest')
    median_index = stdevs.index(median)
    median_dist = off_dicts[median_index]
    mses = meansquarederror(median_dist,parameters[median_index])

    #plot_kde(median_dist,note="(run with median variance)")

    off_errors = [[abs(off_runs[i][0][steps-1][p]-parameters[i][p]) 
                  for i in range(runs)] for p in range(1)]  

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
    
    median_T1 = np.median(T1s)
    print("----------------")
    print("Median results: ")
    print("- T1 = %.4f" % median_T1)
    stdev = off_stdevs[0][steps]
    print("- Variance: %d (σ=%.1f)\n- MSE: %.1f (sqrt=%.1f)\n- Deviation: %.1f"
                    % (stdev**2, stdev,mses[0],mses[0]**0.5, off_error[0]))
    
    fig, axs = plt.subplots(1,figsize=(12,8))

    p=0
    x1 = np.array([i for i in range(steps+1)])
    oy1 = np.array([off_stdevs[p][i] for i in range(steps+1)])
    axs.set_ylabel(r'$\sigma$')
    axs.plot(x1, oy1, color='red', label='offline estimation')
    oq11 = np.array([off_stdevs_q1s[p][i] for i in range(steps+1)])
    oq31 = np.array([off_stdevs_q3s[p][i] for i in range(steps+1)])
    axs.fill_between(x1,oq11,oq31,alpha=0.1,color='red')
    axs.set_title('T1 estimation')
    axs.set_xlabel('Iteration number')
    axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
def print_stats(runs,steps):
    print("* Average number of resampler calls: %d (%d%%)." 
          % (resampler_calls/runs,round(100*resampler_calls/(runs*steps))))
    if (total_HMC!=0 or total_MH!=0):
        print("* Percentage of HMC steps:  %.1f%%." 
              % (100*total_HMC/(total_HMC+total_MH)))
    if (total_HMC != 0):
        print("* Hamiltonian Monte Carlo: %d%% mean particle acceptance rate." 
              % round(100*accepted_HMC/total_HMC))
    if (total_MH != 0):
        print("* Metropolis-Hastings:     %d%% mean particle acceptance rate." 
          % round(100*accepted_MH/total_MH))
    print("(n=%d; runs=%d; 2d)" % (N_particles,runs))

def uniform_prior():
    f_min = left_boundaries[0]
    f_max = right_boundaries[0]
    fs = np.linspace(f_min,f_max,N_particles)
    prior = {}
    for f in fs:
        particle = np.array([f])
        # Use bit strings as keys because numpy arrays are not hashable.
        key = particle.tobytes()
        prior[key] = 1/N_particles
    return prior

def offline_estimation(distribution, data,
                       plot=False):
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
    plot: bool, optional
        Whether to plot the distribution a few times throughout (Default is 
        False).
        
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
    # We'll use evenly spaced times, fixed in advance.
    current_mean, current_stdev = SMCparameters(distribution)
    means, stdevs = [], []
    means.append(current_mean)
    stdevs.append(current_stdev) 

    updates = len(data)
    if updates < 10:
        progress_interval = 100/updates
    print("|0%",end="|"); counter = 0
    resampler_calls = 0
    for i in range(updates):
        if plot and i%(updates/10)==0: # Generate 10 distribution plots.
            info = "- step %d" % (i)
            info += " (resampled %d time(s))" % resampler_calls
            resampler_calls = 0
            plot_kde(distribution,note=info)

        # Update the distribution: get the posterior of the current iteration, 
        #which is the prior for the next.
        distribution,resampled = bayes_update(data[:(i+1)], distribution) 
        resampler_calls += 1 if resampled else 0

        current_mean, current_stdev = SMCparameters(distribution)
        means.append(current_mean)
        stdevs.append(current_stdev) 

        if updates < 10:
            counter+=progress_interval
            print(round(counter),"%",sep="",end="|")
        elif (i%(updates/10)<1): 
            counter+=10
            print(counter,"%",sep="",end="|")
    print("")
    if plot:
        plot_kde(distribution,note=" (final distribution)")
    return distribution, (means, stdevs, _)

def get_data(upload=False, filename=None, steps=75, tmin=1, tmax=3.5,rev=False,
             rep=1, rand=False):
    if upload:
        print("> Real times are approximations only (real data).")
        with open(filename, 'rb') as filehandle: #3, 10
            print("> Uploading data from file \'%s\'..." % filename)
            data = pickle.load(filehandle)
            # Overwrite step number and maximum time.
            steps, tmax = len(data), max([t for t,outcome in data])
            # Flip the outcomes because the code is structured oppositely to
            #the IBM experiments.
            steps, tmax = len(data), max([t for t,outcome in data])
            if rev:
                data = data[::-1]
            #data = data[::20]
            data = [datum for datum in data if datum[0]<40]
    else:
        if rand:
            ts = [random.uniform(tmin,tmax) for i in range(steps)]
        else:
            ts = np.linspace(tmin,tmax,steps)
            ts = np.repeat(ts,rep)
            #ts = np.concatenate([ts,np.linspace(0,10,100)])
            steps=len(ts)
            if rev:
                ts = ts[::-1]
        print(("> Using randomly generated " if rand else 
              "> Using evenly spaced ordered ") + "times.")
        data = [(t,measure(t)) for t in ts]
        
    print(data)
    print("> Tmax = %.1f; steps = %d; T1max = %.1f." 
          % (tmax,steps,*right_boundaries))
    return data

def main():
    global T1_real, N_particles, right_boundaries, dim
    T1_max = right_boundaries[0]
    T1_real = 67
    print("> T1 = %.2f" % T1_real)
    
    unique_ts = 75
    filename_start = 'guadalupe_amplitude_damping_data[2.0,49[T1_est=49_sched=75_nshots=20'
    ndatasets = 10
    datasets = []
    for i in range(ndatasets):
        filename = filename_start + ('_%d.data' % i)
        data = get_data(filename=filename, upload=True, steps=unique_ts, 
                        tmin=1, tmax=70, rev=True, rep=20, rand=False)
        datasets.append(data)
        
    runs_each = 10
    runs = ndatasets*runs_each
    prior = uniform_prior()
    print("> Using %d datasets for %d runs each." % (ndatasets, runs_each))
    
    off_runs = []
    
    parameters = [np.array([T1_real]) for i in range(runs)]
    T1s = []
    off_dicts = []
    try:
          for i in range(ndatasets):
              print(f"> Run {i}.")
              data = datasets[i]
              for j in range(runs_each):
                  (dist,sequences) = offline_estimation(copy.deepcopy(prior),
                                                        data,plot=False)
                  off_runs.append(sequences)
                  off_dicts.append(dist)
                  steps = len(sequences[0])-1
                  T1 = sequences[0][steps]
                  T1s.append(T1)
                  print("> Estimated T1=%.2f." % T1)
    except KeyboardInterrupt:
          e = sys.exc_info()[0]
          runs = len(T1s)
          print(" Quit at run %d (%s)." % (runs,e))

    if runs!=0:
        show_results(off_runs,off_dicts,T1s,parameters)
        
        print_stats(runs,steps)
        
main()