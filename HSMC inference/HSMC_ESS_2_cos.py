# -*- coding: utf-8 -*-
"""
Performs inference for the frequencies of a probability function given by
a product of squared cosines.

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

import sys, itertools, random, matplotlib.pyplot as plt
from autograd import grad, numpy as np
np.seterr(all='warn')
dim=2
total_HMC, accepted_HMC = 0, 0
total_MH, accepted_MH = 0, 0

N_particles = 100 # Number of samples used to represent the probability
#distribution, using a sequential Monte Carlo approximation.

real_parameters = None

lbound = np.array([0,0]) # The left boundaries for the parameters.
rbound = np.array([1,1]) # The right boundaries for the parameters.

def measure(t):
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
        
    Returns
    -------
    1 if the result is |+>, 0 if it is |->.
    '''
    global real_parameters
    r = random.random()
    p = np.random.binomial(1, p=np.cos(real_parameters[0]*t/2)**2*
                           np.cos(real_parameters[1]*t/2)**2)
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
    p=np.cos(particle[0]*t/2)**2*np.cos(particle[1]*t/2)**2
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
    if autograd:
        DU_f = grad(target_U,1)
        DU = np.array(DU_f(data,particle))
    else:
        DU = np.array([0.,0.])
        for (t,outcome) in data:
            f = np.cos(particle[0]*t/2)**2*np.cos(particle[1]*t/2)**2
            if outcome==1:
                DU+=np.array([t*np.sin(particle[0]*t/2)*
                              np.cos(particle[0]*t/2)*
                              np.cos(particle[1]*t/2)**2/f,
                              t*np.sin(particle[1]*t/2)*
                              np.cos(particle[1]*t/2)*
                              np.cos(particle[0]*t/2)**2/f])
            if outcome==0:
                DU+=np.array([-t*np.sin(particle[0]*t/2)*
                              np.cos(particle[0]*t/2)*
                              np.cos(particle[1]*t/2)**2/(1-f),
                              -t*np.sin(particle[1]*t/2)*
                              np.cos(particle[1]*t/2)*
                              np.cos(particle[0]*t/2)**2/(1-f)])
    return(DU)

'''
Test the differentiation:
data = np.array([(1.7,1),(2.4,0)])
particle = np.array([0.3,0.9])
print("Autodiff: ", U_gradient(data,particle,autograd=True))
print("Analytical: ",U_gradient(data,particle,autograd=False))
'''

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
                             left_constraints=lbound, 
                             right_constraints=rbound):
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
                      left_constraints=lbound, 
                      right_constraints=rbound):  
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
        

# Seems to work nice for: M=I, L=10, eta=0.001, N_particles=100, 
#resampling at N_particles/2. Mostly tested for frequencies 0.2 and 0.7. 
first_hamiltonian_MC_step = True
def hamiltonian_MC_step(data, particle, 
                        M=np.identity(2), L=10, eta=0.001, threshold=0.1):
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

def bayes_update(data, distribution, threshold=N_particles/2):
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
    threshold: float, optional
        The threshold effective sample size that should trigger a resampling 
        step (Default is N_particles/2). 
    '''
    global N_particles
    acc_weight, acc_squared_weight = 0, 0
    
    # Perform a correction step by re-weighting the particles according to 
    #the last datum obtained.
    for key in distribution:
        particle = np.frombuffer(key,dtype='float64')
        new_weight = likelihood(data[-1],particle)*distribution[key]
        distribution[key] = new_weight
        acc_weight += new_weight
    
    # Normalize the weights.
    for key in distribution:
        w = distribution[key]/acc_weight
        distribution[key] = w
        acc_squared_weight += w**2 # The inverse participation ratio will be
        #used to decide whether to resample.

    if (1/acc_squared_weight <= threshold):
        # Perform an importance sampling step (with replacement) according to                
        #the updated weights.
        selected_particles = random.choices(list(distribution.keys()), 
                                              weights=distribution.values(),
                                              k=N_particles)
        
        stdevs = SMCparameters(selected_particles,list=True)[1]
        Cov = np.diag(stdevs**2)
        #Cov = np.identity(2) # To test just identity mass. Fix SMC for above
        
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
            
    return distribution

def SMCparameters(distribution, stdev=True, list=False):
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
    means, meansquares = np.zeros(dim)
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

first_offline_estimation = True
def offline_estimation(distribution, steps, increment=0.08):
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
        print("Offline estimation: (9/8)^k")
        first_offline_estimation = False
        
    # We'll use evenly spaced times, fixed in advance.
    current_mean, current_stdev = SMCparameters(distribution)
    means, stdevs = [], []
    means.append(current_mean)
    stdevs.append(current_stdev) 
    data = []
    
    #ts = np.arange(1, steps+1)*increment
    ts = [(9/8)**k for k in range(steps)]
    for i,t in enumerate(ts):
        data.append((t,measure(t)))
        # Update the distribution: get the posterior of the current iteration, 
        #which is the prior for the next.
        if (i<0):
          distribution = bayes_update(data, distribution,threshold=0) 
        else:
          distribution = bayes_update(data, distribution) 
        
    return distribution

def main():
    global real_parameters, N_particles, lbound, rbound, dim
    real_parameters = np.array([0.2,0.7])

    # We will start with a uniform prior within the boundaries. 
    
    each = N_particles**(1/dim) # Number of particles along each dimension.
    width = [rbound[i]-lbound[i] for i in range(dim)] # Parameter spans.
    
    # Create evenly spaced lists within the region defined for each parameter.
    particles = [np.arange(lbound[i]+width[i]/each,rbound[i],
                   width[i]/each) for i in range(dim)]
    
    # Form list of all possible tuples (outer product of the lists for each 
    #parameter).
    particles = list(itertools.product(*particles))
    
    prior = {}
    for particle in particles:
        particle = np.asarray(particle)
        # Use bit strings as keys because numpy arrays are not hashable.
        key = particle.tobytes()
        prior[key] = 1/N_particles
    
    steps = 50
    final_dist = offline_estimation(prior.copy(),steps)
    
    keys = list(final_dist.keys())
    particles = [np.frombuffer(key,dtype='float64') for key in keys]
    
    fig, axs = plt.subplots(1,figsize=(8,8))

    plt.xlim([0,1])
    plt.ylim([0,1])
    
    x1 = [particle[0] for particle in particles]
    x2 = [particle[1] for particle in particles]
    weights = [final_dist[key]*200 for key in keys]
    axs.scatter(x1, x2, marker='o',s=weights)

    print("n=%d; N=%d" % (N_particles,steps))
    
    global total_HMC, accepted_HMC, total_MH, accepted_MH
    if (total_HMC != 0) or (total_MH != 0):
      print("* Total resampler calls:  %d." 
            % ((total_HMC+total_MH)/N_particles))
      print("* Percentage of HMC steps:  %.1f%%." 
            % (100*total_HMC/(total_HMC+total_MH)))
    if (total_HMC != 0):
        print("* Hamiltonian Monte Carlo: %d%% mean particle acceptance rate." 
              % round(100*accepted_HMC/total_HMC))
    if (total_MH != 0):
        print("* Metropolis-Hastings:     %d%% mean particle acceptance rate." 
              % round(100*accepted_MH/total_MH))

    
main()