# -*- coding: utf-8 -*-
"""
Module with SMC resampler-related functions.
"""
import random, numpy as np
import global_vars as glob
from modules.likelihoods import likelihood, target_U, U_gradient
from modules.distributions import SMCparameters

first_metropolis_hastings_step = True
first_hamiltonian_MC_step = True

total_HMC, accepted_HMC = 0, 0
total_MH, accepted_MH = 0, 0

total_HMC, accepted_HMC = 0, 0
total_MH, accepted_MH = 0, 0

def init_resampler(print_info=True):
    '''
    Initializes some counters respecting the Markov transitions used by the 
    resampler, as well as some constants pertaining to the module.
    
    Parameters
    ----------
    print_info: bool, optional
        Whether to print the resampler settings in use.
    '''
    if print_info:
        global first_hamiltonian_MC_step, first_metropolis_hastings_step
        first_hamiltonian_MC_step = True
        first_metropolis_hastings_step = True
        first_pseudomarginal_MH = True
    
    global total_HMC, accepted_HMC, total_MH, accepted_MH, total_u, accepted_u
    total_HMC, accepted_HMC = 0, 0
    total_MH, accepted_MH = 0, 0
    
first_metropolis_hastings_step = True
def metropolis_hastings_step(data, particle, S=None,
                             factor=0.001):
    '''
    Performs a Metropolis-Hastings mutation on a given particle, using a 
    gaussian function for the proposals.
    
    Parameters
    ----------
    data: [([float],int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    particle: [float]
        The particle to undergo a mutation step.
    S: [[float]]
        The covariance matrix that will determine the standard deviations
        of the normal distributions used for the proposal in each dimension
        (the dth diagonal element corresponding to dimension d).
    factor: float, optional
        The factor M is to be multiplied by to get the actual standard 
        deviations (Default is 0.01).
        
    Returns
    -------
    particle: [float]
        The mutated particle.
    p: float
        The acceptance probability to be used for the evolved particle as a 
        Monte Carlo proposal.
    '''
    dim, left_constraints, right_constraints = \
        glob.dim, glob.lbound, glob.rbound

    global first_metropolis_hastings_step
    if first_metropolis_hastings_step:
      if not np.array_equal(S,np.identity(dim)):
          Cov = "Cov"
      else:
          Cov = "I"
      print("MH:  S=%s, factor=%.4f" % (Cov,factor))
      first_metropolis_hastings_step = False

    S = np.identity(dim)
    Sigma = factor*S
    # Start with any invalid point.
    new_particle = np.array([left_constraints[i]-1 for i in range(dim)]) 
    
    # Get a proposal that satisfies the constraints.
    while any([new_particle[i]<left_constraints[i] for i in range(dim)] + 
                  [new_particle[i]>right_constraints[i] for i in range(dim)]):
        new_particle = np.array([np.random.normal(particle[i], Sigma[i][i])
                                 for i in range(dim)])

    # The proposal function is symmetric with respect to the particles.
    p = likelihood(data,new_particle)/likelihood(data,particle)
    return new_particle,p
    
def simulate_dynamics(data, initial_momentum, initial_particle, M,L,eta):  
    '''
    Simulates Hamiltonian dynamics for a given particle, using leapfrog 
    integration.
    
    Parameters
    ----------
    data: [([float],int)]
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
        
    Returns
    -------
    particle: [float]
        The particle having undergone motion.
    p: float
        The acceptance probability to be used for the evolved particle as a 
        Monte Carlo proposal.
    '''
    dim, left_constraints, right_constraints = \
        glob.dim, glob.lbound, glob.rbound
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
    '''
    if (p<0.1):
        sys.exit('p too small')
    '''
    return new_particle, p
        
first_hamiltonian_MC_step = True
def hamiltonian_MC_step(data, particle, 
                       M=None, L=10, eta=0.01, s=0.1,threshold=0.1):
    '''
    Performs a Hamiltonian Monte Carlo mutation on a given particle.
    
    Parameters
    ----------
    data: [([float],int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    particle: [float]
        The particle to undergo a mutation step.
    M: [[float]], optional
        The mass matrix/Euclidean metric to be used when simulating the
        Hamiltonian dynamics (a HMC tuning parameter) (Default is None, the 
        identity matrix will be used).
    L: int, optional
        The amount of integration steps to be used when simulating the 
        Hamiltonian dynamics (a HMC tuning parameter) (Default is 10).
    eta: float, optional
        The mean integration stepsize to be used when simulating the Hamiltonian 
        dynamics (a HMC tuning parameter) (Default is 0.01).
    s: float, optional
        Some standard deviation by which to perturb the stepsize, which will be
        drawn from a normal distribution with mean `eta` and variance 
        (`s`*`eta`)^2 (Default is 0.1).
    threshold: float, optional
        The highest HMC acceptance rate that should trigger a Metropolis-
        -Hastings mutation step (as an alternative to a  HMC mutation step) 
        (Default is 0.1). 
        
    Returns
    -------
    particle: [float]
        The mutated particle.
    '''
    dim = glob.dim
    if M is None:
        M=np.identity(glob.dim)
    if (threshold<1):
        # Perform a Hamiltonian Monte Carlo mutation.
        global first_hamiltonian_MC_step
        if first_hamiltonian_MC_step:
            if not np.array_equal(M,np.identity(dim)):
                mass = "Cov^-1"
            else:
                mass = "I"
            print("HMC: %s, L=%d, eta=%.10f*N(1,s=%f), threshold for RWM = %.2f" 
                  % (mass,L,eta,s,threshold))
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

    eta = eta*np.random.normal(1,scale=s)
    
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
        
def HMC_resampler(data, distribution, allow_repeated):
    '''
    Importance samples the particles of a distribution, then perturbs each 
    particle (performing Markov moves).
    
    Parameters
    ----------
    data: [([float],int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The prior distribution (SMC approximation).
    allow_repeated: bool
        Whether to allow repeated particles. If False, extra Markov steps will 
        be taken to ensure non-repetition when necessary, and alternative 
        mutations will be used whem HMC is unlikely to get an accepted proposal.
        
    Returns
    -------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The resampled distribution (SMC approximation).
    success: bool
        Whether the resampling has been successful; it fails if the current 
        covariance matrix is not invertible.
    '''
    N_particles = glob.N_particles
    mean,stdevs = SMCparameters(distribution)
    
    # Perform an importance sampling step (with replacement) according to                
    #the updated weights.
    selected_particles = random.choices(list(distribution.keys()), 
                                          weights=distribution.values(),
                                          k=N_particles)
                                          
    stdevs = SMCparameters(selected_particles,list=True)[1]
    Cov = np.diag(stdevs**2)
    
    # Check for singularity (the covariance matrix must be invertible).
    success = True
    if (np.linalg.det(Cov) == 0): 
        success = False
        print("\n> The covariance matrix must be invertible, but it is\n%s"
              "\n...As such, the updating process has been interrupted."
              " [resample]" % Cov)
        return distribution, success
    
    Cov_inv = np.linalg.inv(Cov)

    # Perform a mutation step on each selected particle, eventually imposing 
    #that the particles be unique.
    distribution.clear()
    for key in selected_particles:
        repeated = True
        while repeated:
            particle = np.frombuffer(key,dtype='float64')
            thr = 0 if allow_repeated else 0.1
            mutated_particle = hamiltonian_MC_step(data,particle,
                                                  M=Cov_inv, threshold=thr)
            key = mutated_particle.tobytes()
            if key not in distribution:
                repeated = False
            elif allow_repeated:
                # Key alreay exists.
                distribution[key] += 1/N_particles
                break
        if not repeated:
            # Just create particle in either case.
            distribution[key] = 1/N_particles
    
    return distribution, success
    
def HMC_resampler_stats():
    '''
    Provides resampler related statistics, referring to since the moment 
    `init_resampler` is called. 
    
    Returns
    -------
    resampler_calls: int
        The total amount of times the resampler was called on the distribution.
    acceptance_ratio: int
        The percentual acceptance ratio for the HMC proposals.
    '''
    N_particles = glob.N_particles
    resampler_calls, acceptance_ratio = 0, None
    if (total_HMC != 0):
        resampler_calls = int(total_HMC/N_particles)
        acceptance_ratio = int(round(100*accepted_HMC/total_HMC))
    return(resampler_calls,acceptance_ratio)
    
    
def print_resampler_stats():
    '''
    Prints some relevant information relative to the run of the resampler on
    the console, counting from the moment `init_resampler` is called.
    '''
    N_particles = glob.N_particles
    if (total_HMC != 0) or (total_MH != 0):
      print("* Total resampler calls:  %d." 
            % ((total_HMC+total_MH)/N_particles))
      print("* Percentage of HMC steps:  %.1f%%." 
            % (100*total_HMC/(total_HMC+total_MH)))
    if (total_HMC != 0):
        print("* Hamiltonian Monte Carlo: %d%% mean particle acceptance rate." 
              % round(100*accepted_HMC/total_HMC))
    if (total_MH != 0):
        print("* Metropolis-Hastings: %d%% mean particle acceptance rate." 
              % round(100*accepted_MH/total_MH))