# -*- coding: utf-8 -*-
"""
Module with SMC resampler-related functions.
"""
import random, numpy as np
import global_vars as glob
from function_evaluations.likelihoods import likelihood, target_U, U_gradient
from function_evaluations.estimators import Taylor_coefs,likelihood_estimator,\
    loggradient_estimator
from tools.statistics import gaussian, SMCparameters

first_metropolis_hastings_step = True
first_hamiltonian_MC_step = True
first_pseudomarginal_MH = True
first_HMC_resampler = True

total_HMC, accepted_HMC = 0, 0
total_MH, accepted_MH = 0, 0
total_u, accepted_u = 0, 0

def init_resampler():
    '''
    Initializes some counters respecting the Markov transitions used by the 
    resampler, as well as some constants pertaining to the module.
    '''
    global first_hamiltonian_MC_step, first_metropolis_hastings_step, \
        first_pseudomarginal_MH
    first_hamiltonian_MC_step = True
    first_metropolis_hastings_step = True
    first_pseudomarginal_MH = True
    first_HMC_resampler = True
    
    global total_HMC, accepted_HMC, total_MH, accepted_MH, total_u, accepted_u
    total_HMC, accepted_HMC = 0, 0
    total_MH, accepted_MH = 0, 0
    total_u, accepted_u = 0, 0
    

def metropolis_hastings_step(data, subs_info, particle, coef, 
                                 Cov=None, factor=0.05):
    '''
    Performs a Metropolis-Hastings mutation on a given particle, using a 
    gaussian function for the proposals.
    
    Parameters
    ----------
    data: [([float],int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    subs_info: ([int],[float],(float,float,float),bool)
        A tuple containing all required subsampling-related information: the
        subsampling indices, the mean/center for the Taylor expansions (which 
        should approximate the distribution location; may be a dummy value if 
        not using control variates), the coefficients for said expansions 
        (again irrelevant if not using control variates), and a boolean 
        indicating whether to use control variates, by this order.
    particle: [float]
        The particle to undergo a mutation step.
    coef: float
        The tempering coefficient to be used in computing the (annealed)
        likelihood.
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
        
    #Cov = np.identity(glob.dim)
    
    global first_metropolis_hastings_step
    if first_metropolis_hastings_step:
      if not np.array_equal(Cov,np.identity(dim)):
          Sigma = "Sigma**0.5"
      else:
          Sigma = "I"
      print("MH:  Sigma=%s, factor=%.4f" % (Sigma,factor))
      first_metropolis_hastings_step = False

    
    Sigma = factor*Cov**0.25 # Numpy takes stdev for gaussians, not variance.
    # Start with any invalid point.
    new_particle = np.array([left_constraints[i]-1 for i in range(dim)]) 
    
    # Get a proposal that satisfies the constraints.
    while any([new_particle[i]<left_constraints[i] for i in range(dim)] + 
                  [new_particle[i]>right_constraints[i] 
                   for i in range(dim)]):
         new_particle = np.array([np.random.normal(particle[i], Sigma[i][i])
                                  for i in range(dim)])
    if subs_info is None:
        p = (likelihood(data,new_particle,coef=coef)/
                likelihood(data,particle,coef=coef))
    else:
        indices, mean, Tcoefs, control_variates = subs_info
        old_L = likelihood_estimator(data, particle, coef, indices, mean, 
                                     Tcoefs,control_variates=control_variates)
        new_L = likelihood_estimator(data, new_particle, coef, indices, mean, 
                                     Tcoefs,control_variates=control_variates)
        p = 1 if old_L==0 else new_L/old_L
    return new_particle,p
    
def simulate_dynamics(data,subs_info, coef, initial_momentum, initial_particle, 
                      M,L,eta):  
    '''
    Simulates Hamiltonian dynamics for a given particle, using leapfrog 
    integration.
    
    Parameters
    ----------
    data: [([float],int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    subs_info: ([int],[float],(float,float,float),bool)
        A tuple containing all required subsampling-related information: the
        subsampling indices, the mean/center for the Taylor expansions (which 
        should approximate the distribution location; may be a dummy value if 
        not using control variates), the coefficients for said expansions 
        (again irrelevant if not using control variates), and a boolean 
        indicating whether to use control variates, by this order.
    coef: float
        The tempering coefficient to be used in computing the (annealed)
        likelihood.
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

    if subs_info is None:
        DU = U_gradient(data,initial_particle,coef)
    else:
        indices, mean, Tcoefs, control_variates = subs_info
        DU = -loggradient_estimator(data,initial_particle,coef,indices,mean,
                                    Tcoefs,control_variates=control_variates)
    
    # Perform leapfrog integration according to Hamilton's equations.
    new_particle = initial_particle
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

        if subs_info is None:
            DU = U_gradient(data,new_particle,coef)
        else:
            DU = -loggradient_estimator(data,new_particle,coef,indices,
                              mean,Tcoefs,control_variates=control_variates)
        if (l != L-1):
            new_momentum = np.add(new_momentum,-eta*DU)     
    new_momentum = np.add(new_momentum,-0.5*eta*DU)
    
    # Compute the acceptance probability.
    if subs_info is None:
        p = np.exp(target_U(data,initial_particle,coef)
                   -target_U(data,new_particle,coef)
                   +np.sum(np.linalg.multi_dot(
                       [initial_momentum,M_inv,initial_momentum]))/2
                   -np.sum(np.linalg.multi_dot(
                       [new_momentum,M_inv,new_momentum]))/2)
    else: 
        prev_U = -np.log(likelihood_estimator(data, initial_particle, coef, 
                                      indices, mean, Tcoefs,
                                      control_variates=control_variates))
        new_U = -np.log(likelihood_estimator(data, new_particle, coef, indices, 
                                     mean, Tcoefs,
                                     control_variates=control_variates))
        p = np.exp(prev_U-new_U
                   +np.sum(np.linalg.multi_dot(
                       [initial_momentum,M_inv,initial_momentum]))/2
                   -np.sum(np.linalg.multi_dot(
                       [new_momentum,M_inv,new_momentum]))/2)
    '''
    if (p<0.1):
        sys.exit('p too small')
    '''
    return new_particle, p
        
def hamiltonian_MC_step(data, particle,coef,
                    M=None, L=20, eta=0.001, threshold=0.1,
                    subs_info=None):
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
    coef: float
        The tempering coefficient to be used in computing the (annealed)
        likelihood.
    M: [[float]], optional
        The mass matrix/Euclidean metric to be used when simulating the
        Hamiltonian dynamics (a HMC tuning parameter) (Default is the identity
        matrix).
    L: int, optional
        The amount of integration steps to be used when simulating the 
        Hamiltonian dynamics (a HMC tuning parameter) (Default is 10).
    eta: float, optional
        The integration stepsize to be used when simulating the Hamiltonian 
        dynamics (a HMC tuning parameter) (Default is 0.01).
    threshold: float, optional
        The highest HMC acceptance rate that should trigger a Metropolis-
        -Hastings mutation step (as an alternative to a  HMC mutation step) 
        (Default is 0.1). 
    subs_info: ([int],[float],(float,float,float),bool), optional
        A tuple containing all required subsampling-related information 
        (Default is None). This comprises the subsampling indices, the mean/
        center for the Taylor expansions (which should approximate the 
        distribution location; may be a dummy value if not using control 
        variates), the coefficients for said expansions (again irrelevant if 
        not using control variates), and a boolean indicating whether to use 
        control variates, by this order.
        
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
            print("HMC: %s, L=%d, eta=%.10f, RWM-thresh=%.2f" % 
                    (mass,L,eta,threshold))
            first_hamiltonian_MC_step = False
            
        global total_HMC, accepted_HMC, total_MH, accepted_MH
        initial_momentum = np.random.multivariate_normal(np.zeros(dim), M)
        new_particle, p = \
            simulate_dynamics(data,subs_info,coef,initial_momentum,particle,
                              M,L,eta)
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
        new_particle,p = metropolis_hastings_step(data,subs_info,particle,coef,
                                                  Cov=Cov)
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

def pseudomarginal_MH(data, particle, indices, coef, mean, Tcoefs,
                      control_variates, blocks=1):
    '''
    Picks the next subsampling indices (auxiliary variables in a pseudo-
    -marginal approach) by performing a Metropolis-Hastings transition. 
    
    Parameters
    ----------
    data: [([float],int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    particle: [float]
        The set of parameters at which the loglikelihood is to be evaluated.
    indices: [float]
        The subsampling indices. 
    coef: float OR [float,float]
        The tempering coefficient (or a pair of, for the re-weighting ratio) 
        for the likelihood.
    mean: [float]
        The mean particle, which defines the point around which the Taylor 
        expansion will be evaluated.
    Tcoefs: (float,float,float) 
        The 0-th to 2nd order Taylor coefficients.
    control_variates: bool
        Whether to use control variates in the estimator.
    blocks: int, optional
        The number of blocks to split the subsampling indices into (Default is 
        7). Of these, a random one will be updated (attributed new indices).
        The purpose is to induce correlation between consecutive proposals,
        thereby reducing the variability in their likelihood estimators and 
        yielding a more robust acceptance ratio.
        
    Returns
    -------
    indices: [int]
        The (tempered) likelihood estimator.
    '''
    global first_pseudomarginal_MH
    if first_pseudomarginal_MH is True:
        print(("Block pseudo marginal Metropolis-Hastings: %d block(s) "
               +"(induced correlation roughly %.1f%%)") % 
              (blocks,(1-1/blocks)*100))
        first_pseudomarginal_MH = False
        
    measurements, samples = glob.measurements, glob.samples
    global total_u, accepted_u

    # Choose the indices of the subsampling indices to be changed in the 
    #proposal.
    chosen_block = random.randrange(0,blocks)
    block_length = samples//blocks
    block_indices = [chosen_block*block_length+i for i in range(block_length)]
    remaining = samples%blocks
    if remaining!=0 and chosen_block==blocks-1:
        # Append all leftover indices to last block.
        block_indices.extend([blocks*block_length+i for i in range(remaining)])

    new_indices = indices.copy()
    for index in block_indices:
        new_indices[index] = random.randrange(0,measurements)

    # Calculate the previous and new likelihood estimators for the acceptance
    #probability.
    prev_l = likelihood_estimator(data,particle,coef,indices,mean,Tcoefs,
                                  control_variates=control_variates)
    new_l = likelihood_estimator(data,particle,coef,new_indices,mean,Tcoefs,
                                 control_variates=control_variates)
    p = new_l/prev_l
    total_u += 1
    a = min(1,p)
    if (np.random.rand() < a):
        accepted_u += 1
        indices = new_indices
    return (indices)

def gibbs_update(data, particle, indices, coef, M, mean, Tcoefs,
                 control_variates,HMCthreshold):
    '''
    Picks the next sample (parameter vector + subsampling indices) by 
    performing a Gibbs update.
    First the parameters are held fixed and the indices are picked 
    conditionally on them (using Metropolis-Hastings), then these fresh indices
    are held fixed and the parameters are picked conditionally on them (using
    Hamiltonian Monte Carlo). 
    
    Parameters
    ----------
    data: [([float],int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    particle: [float]
        The set of parameters at which the loglikelihood is to be evaluated.
    indices: [float]
        The subsampling indices. 
    coef: float OR [float,float]
        The tempering coefficient (or a pair of, for the re-weighting ratio) 
        for the likelihood.
    mean: [float]
        The mean particle, which defines the point around which the Taylor 
        expansion will be evaluated.
    Tcoefs: (float,float,float) 
        The 0-th to 2nd order Taylor coefficients.
    control_variates: bool
        Whether to use control variates in the estimator.
        
    Returns
    -------
    new_particle: [float]
        The updated parameter vector.
    new_indices: [int]
        The updated subsampling indices.
    '''
    new_indices = pseudomarginal_MH(data,particle,indices, coef, mean,Tcoefs,
                                    control_variates)
    
    info = (indices, mean, Tcoefs, control_variates)
    new_particle = hamiltonian_MC_step(data,particle,coef,M=M,
                                       subs_info=info,threshold=HMCthreshold)
    return(new_particle,new_indices)

def HMC_resampler(data, distribution, coef, subsample, control_variates,HMCthreshold=0.1,moves=1):
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
    coef: float
        The tempering coefficient corresponding to the current iteration (i.e.
        the one determining the target distribution of the current SMC step).
    previous_coef: float
        The tempering coefficient corresponding to the previous iteration (i.e.
        the one that determined the target distribution of the preceding SMC 
        step, which is approximated by the pre-update particle cloud given by
        `distribution`).
    subsample: bool
        Whether to subsample.
    control_variates: bool
        Whether to use control variates (Taylor approximations of each datum's
        likelihood, yielding a difference estimator with a smaller variance).
    HMCthreshold: float, optional
        The threshold acceptance ratio that should trigger a random walk 
        Metropolis step in place of the HMC step (Default is 0.1).
    moves: int, optional
        The number of Markov moves to be performed per particle (Default is 1).
        
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
    global first_HMC_resampler
    if first_HMC_resampler is True:
        print("SMC resampler: %d Markov move(s) " % moves)
        first_HMC_resampler = False
        
    N_particles = glob.N_particles
    mean,stdevs = SMCparameters(distribution)
    Tcoefs = Taylor_coefs(data,mean)
    
    # Perform an importance sampling step (with replacement) according to                
    #the updated weights.
    values = list(distribution.values())
    importances = [v[0] for v in values]
    selected_particles = random.choices(list(distribution.keys()), 
                                          weights=importances,
                                          k=N_particles)
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

    # Perform a mutation step on each selected particle, imposing that the
    #particles be unique.
    new_distribution = {}
    for key in selected_particles:
        particle = np.frombuffer(key,dtype='float64')
        u = distribution[key][1]
        repeated = True
        while (repeated == True):
            for m in range(moves):
                if subsample:
                    particle, u = gibbs_update(data, particle, u,coef, 
                                            Cov_inv,mean,Tcoefs,control_variates,HMCthreshold)
                else: 
                    particle = hamiltonian_MC_step(data,particle,coef,
                                                       M=Cov_inv, threshold=HMCthreshold)
                    # Don't update u, won't be used.
            new_key = particle.tobytes()
            if new_key not in new_distribution: 
                repeated = False
                
        new_distribution[new_key] = [1/N_particles,u]
        
    distribution.clear()
    distribution = new_distribution
    
    return distribution, success

def print_resampler_stats(runs=1):
    '''
    Prints some relevant information relative to the run of the resampler on
    the console, counting from the moment `init_resampler` is called.
    '''
    N_particles = glob.N_particles
    if (total_HMC != 0) or (total_MH != 0):
      print("* Total resampler calls (average per run):  %d." 
            % ((total_HMC+total_MH)/(N_particles*runs)))
      print("* Percentage of HMC steps:  %.1f%%." 
            % (100*total_HMC/(total_HMC+total_MH)))
    if (total_HMC != 0):
        print("* Hamiltonian Monte Carlo: %d%% mean particle acceptance rate." 
              % round(100*accepted_HMC/total_HMC))
    if (total_MH != 0):
        print("* Metropolis-Hastings: %d%% mean particle acceptance rate." 
              % round(100*accepted_MH/total_MH))
    if (total_u != 0):
        print(("* Metropolis-Hastings (indices): %d%% mean particle acceptance"
              +" rate.") % round(100*accepted_u/total_u))
    init_resampler() # Clear the slate for the next epoch of resampler calls. 