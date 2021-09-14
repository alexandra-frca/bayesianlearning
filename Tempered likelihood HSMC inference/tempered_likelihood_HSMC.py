# -*- coding: utf-8 -*-
"""
Performs inference on the frequencies of a multi-parameter binomial probability
distribution with success probability given by a sum of squared cosines: 
sum_i [ cos(theta_i*t/2)^2 ], using tempered likelihood SMC.

A sequential Monte Carlo approximation is used to represent the probability 
distributions, using Hamiltonian Monte Carlo and Metropolis-Hastings mutation 
steps.

The target distributions consist of a sequence of annealed likelihoods, i.e. 
the likelihood resulting from the whole data set, raised to the power of some 
annealing coefficients.

The particle positions are plotted.
"""

import copy, itertools, random, matplotlib.pyplot as plt
from autograd import grad, numpy as np
np.seterr(all='warn')
dim = 4
total_HMC, accepted_HMC = 0, 0
total_MH, accepted_MH = 0, 0

N_particles = None # Number of samples used to represent the probability
#distribution, using a sequential Monte Carlo approximation.

measurements = None # Length of the data record.

real_parameters = None

lbound = np.zeros(dim) # The left boundaries for the parameters.
rbound = np.ones(dim) # The right boundaries for the parameters.

def measure(t):
    '''
    Samples from the real probability distribution, whose parameters are to be
    inferred.
    
    Parameters
    ----------
    t: float
        The measurement time, an experimental control.
        
    Returns
    -------
    1 or 0 depending on the result.
    '''
    # A sum of squared cosines. 
    r = np.random.binomial(1, p=np.sum(np.cos(real_parameters*t/2)**2/dim))
    return r

def simulate_1(particle, t):
    '''
    Provides an estimate for the likelihood  P(D=1|v,t) of outcome 1 given 
    a vector of parameters v and a time of choice t. 
    
    Parameters
    ----------
    particle: [float]
        The set of parameters to be used in the simulation.
    t: float
        The measurement time, an experimental control.
        
    Returns
    -------
    p: float
        The estimated probability of getting result 1.
    '''
    p=np.sum(np.cos(particle*t/2)**2/dim)
    return p 

def likelihood(data,particle,coef=1):
    '''
    Provides an estimate for the likelihood  P(data|v,t) given a data record, a 
    vector of parameters v and a time of choice t. 
    
    Parameters
    ----------
    data: [([float],int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    particle: [float]
        The set of parameters to be used for the likelihood.
    coef: float
        The tempering coefficient to be used (Default is 1).
        
    Returns
    -------
    p: float
        The annealed likelihood.
    '''
    
    l = np.product([simulate_1(particle,t) if (outcome==1)
                        else 1-simulate_1(particle,t) for (t,outcome) in data])
    l = l**coef
    return l

def target_U(data,particle,coef):
    '''
    Evaluates the target "energy" associated to the joint likelihood of some 
    vector  of data, given a set of parameters. 
    
    Parameters
    ----------
    data: [([float],int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    particle: [float]
        The set of dynamical parameters to be used for the likelihood.
    coef: float
        The tempering coefficient to be used in computing the (annealed)
        likelihood.
        
    Returns
    -------
    U: float
        The value of the "energy".
    '''
    U = -np.log(likelihood(data, particle,coef)) 
    return (U)

def U_gradient(data,particle,coef,autograd=False):
    '''
    Evaluates the gradient of the target "energy" associated to the likelihood 
    at a time t seen as a probability density, given a set of parameters. 
    
    Parameters
    ----------
    data: [([float],int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    particle: [float]
        The set of dynamical parameters to be used for the likelihood.
    coef: float
        The tempering coefficient to be used in computing the (annealed)
        likelihood.
    autograd: bool, optional
        Whether to use automatic differenciation (Default is False).
        
    Returns
    -------
    DU: [float]
        The gradient of the "energy".
    '''
    if autograd:
        DU_f = grad(target_U,1)
        DU = np.array(DU_f(data,particle,coef))
    else:
        DU = np.array(np.zeros(dim))
        for (t,outcome) in data:
            f = np.sum(np.cos(particle*t/2)**2)/dim
            if outcome==1:
                DU+=coef*t/dim*np.sin(particle*t/2)*np.cos(particle*t/2)/f
            if outcome==0:
                DU+=coef*(-t/dim*np.sin(particle*t/2)*np.cos(particle*t/2)
                          /(1-f))
                     
    return(DU)

def test_differentiation():
    '''
    Tests the analytical differentiation in `U_gradient` by comparing its
    numerical result against the one computed by automatic differentiation.
    Both are evaluated for the same representative set of inputs and printed on 
    the console.
    '''
    data = np.array([(random.random(),1),(random.random(),0)])
    particle = np.array([random.random() for d in range(dim)])
    coef = random.random()
    print("Autodiff: ", U_gradient(data,particle,coef,autograd=True))
    print("Analytical: ",U_gradient(data,particle,coef,autograd=False))

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
def metropolis_hastings_step(data, particle, coef, S=None, factor=0.01,
                                 left_constraints=lbound, 
                                 right_constraints=rbound):
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
    left_constraints: [float], optional
        The leftmost bounds to be enforced for the particle's motion (Default 
        is `lbound`).
    right_constraints: [float], optional
        The rightmost bounds to be enforced for the particle's motion (Default 
        is `rbound`).
        
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

    # Compute the probabilities of transition for the acceptance probability.
    inverse_transition_prob = \
        np.product([gaussian(particle[i],new_particle[i],
                                                  Sigma[i][i]) 
                                          for i in range(dim)])
    transition_prob = np.product([gaussian(new_particle[i],particle[i],
                                       Sigma[i][i]) for i in range(dim)])

    p = likelihood(data,new_particle,coef=coef)*inverse_transition_prob/ \
        (likelihood(data,particle,coef=coef)*transition_prob)
    return new_particle,p
    
def simulate_dynamics(data, coef, initial_momentum, initial_particle, M,L,eta,
                      left_constraints=lbound, 
                      right_constraints=rbound):  
    '''
    Simulates Hamiltonian dynamics for a given particle, using leapfrog 
    integration.
    
    Parameters
    ----------
    data: [([float],int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
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
    left_constraints: [float], optional
        The leftmost bounds to be enforced for the particle's motion (Default 
        is `lbound`).
    right_constraints: [float], optional
        The rightmost bounds to be enforced for the particle's motion (Default 
        is `rbound`).
        
    Returns
    -------
    particle: [float]
        The particle having undergone motion.
    p: float
        The acceptance probability to be used for the evolved particle as a 
        Monte Carlo proposal.
    '''
    M_inv = np.linalg.inv(M)
    new_particle = initial_particle
    DU = U_gradient(data,new_particle,coef)
    
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

        DU = U_gradient(data,new_particle,coef)
        if (l != L-1):
            new_momentum = np.add(new_momentum,-eta*DU)     
    new_momentum = np.add(new_momentum,-0.5*eta*DU)
    
    # Compute the acceptance probability.
    p = np.exp(target_U(data,initial_particle,coef)
               -target_U(data,new_particle,coef)
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
def hamiltonian_MC_step(data, particle,coef,
                        M=np.identity(dim), L=10, eta=0.01, threshold=0.1):
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
        
    Returns
    -------
    particle: [float]
        The mutated particle.
    '''
    if (threshold<1):
        # Perform a Hamiltonian Monte Carlo mutation.
        global first_hamiltonian_MC_step
        if first_hamiltonian_MC_step:
            if not np.array_equal(M,np.identity(dim)):
                mass = "Cov^-1"
            else:
                mass = "I"
            print("HMC: %s, L=%d, eta=%.10f" % (mass,L,eta))
            first_hamiltonian_MC_step = False
            
        global total_HMC, accepted_HMC, total_MH, accepted_MH
        initial_momentum = np.random.multivariate_normal(np.zeros(dim), M)
        new_particle, p = \
            simulate_dynamics(data,coef,initial_momentum,particle,M,L,eta)
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
        new_particle, p = metropolis_hastings_step(data,particle,coef,S=Cov)
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

first_bayes_update = True
def bayes_update(data, distribution, coef, previous_coef, threshold,
                 signal_resampling=False):
    '''
    Updates a prior distribution according to the outcome of a measurement, 
    using Bayes' rule. 
    
    Parameters
    ----------
    data: [([float],int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    coef: float
        The tempering coefficient corresponding to the current iteration (i.e.
        the one determining the target distribution of the current SMC step).
    previous_coef: float
        The tempering coefficient corresponding to the previous iteration (i.e.
        the one that determined the target distribution of the preceding SMC 
        step, which is approximated by the pre-update particle cloud given by
        `distribution`).
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The prior distribution (SMC approximation).
    threshold: float
        The threshold effective sample size that should trigger a resampling 
        step. 
    signal_resampling: bool, optional
        Whether to return a second variable denoting the ocurrence of 
        resampling (Default is False).
        
    Returns
    -------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The updated distribution (SMC approximation).
    resampled: bool
        Whether resampling has occurred.
    '''
    global first_bayes_update
    if first_bayes_update is True:
        print("Bayes update: resampling threshold = ", threshold)
        first_bayes_update = False

    global N_particles
    acc_weight, acc_squared_weight = 0, 0
    
    # Perform a correction step by re-weighting the particles according to 
    #the last chunk of data added (i.e. to the ratio between the latest 
    #cumulative likelihood to the previous one, which cancels out all but those 
    #newest data).
    for key in distribution:
        particle = np.frombuffer(key,dtype='float64')
        new_weight = (likelihood(data,particle,
                                 coef=coef-previous_coef)*distribution[key])
        distribution[key] = new_weight
        acc_weight += new_weight
    
    # Normalize the weights.
    for key in distribution:
        w = distribution[key]/acc_weight
        distribution[key] = w
        acc_squared_weight += w**2 # The inverse participation ratio will be
        #used to decide whether to resample.
    
    #print("ESS: ", 1/acc_squared_weight)
    resampled = False
    if (1/acc_squared_weight <= threshold):
        resampled = True
        # Perform an importance sampling step (with replacement) according to                
        #the updated weights.
        selected_particles = random.choices(list(distribution.keys()), 
                                              weights=distribution.values(),
                                              k=N_particles)
        
        stdevs = SMCparameters(selected_particles,list=True)[1]
        Cov = np.diag(stdevs**2)
        
        # Check for singularity (the covariance matrix must be invertible).
        if (np.linalg.det(Cov) == 0): 
            print("\n> The covariance matrix must be invertible, but it is\n%s"
                  "\n...As such, the updating process has been interrupted."
                  " [bayes_update]" % Cov)
            if signal_resampling:
                return distribution, None # Resampling interrupted midway.
            return distribution
    
        Cov_inv = np.linalg.inv(Cov)
    
        # Perform a mutation step on each selected particle, imposing that the
        #particles be unique.
        distribution.clear()
        for key in selected_particles:
            repeated = True
            while (repeated == True):
                particle = np.frombuffer(key,dtype='float64')
                mutated_particle = hamiltonian_MC_step(data,particle,coef,
                                                       M=Cov_inv)
                key = mutated_particle.tobytes()
                if (key not in distribution):
                    repeated = False
            distribution[key] = 1/N_particles
    
    if signal_resampling:
        return distribution, resampled

    return distribution

def SMCparameters(distribution, stdev=True, list=False):
    '''
    Calculates the mean and (optionally) standard deviation of a given 
    distribution.
    
    Parameters
    ----------
    distribution: dict | [float]
        , with (key,value):=(particle,importance weight)
        , or list_item:=particle (respectively)
        The distribution (SMC approximation) whose parameters are to be 
        calculated.
    stdev: bool, optional 
        To be set to False if the standard deviation is not to be returned 
        (Default is True).
    list: bool, optional
        To be set to True if the distribution is given as a list (as opposed to
        a dictionary) (weights are then assumed even).
        
    Returns
    -------
    mean: float
        The mean of the distribution.
    stdev: float
        The standard deviation of the distribution.
    '''
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

first_plot_distribution = True
def plot_distribution(distribution, real_parameters, note=""):
    '''
    Plots a discrete distribution, by scattering points as circles with 
    diameters proportional to their weights. Also signalizes the target modes.
    A graph will be produced for each pair of dimensions.
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The distribution to be plotted (SMC approximation).
    real_parameters: [float]
        The real values of the parameters (the resulting modes will be marked 
         by red 'X's on the graph).
    note: str, optional
        Some string to be appended to the graph title (Default is ""). 
    '''
    n_graphs = dim//2
    for i in range(n_graphs):
        keys = list(distribution.keys())
        particles = [np.frombuffer(key,dtype='float64') for key in keys]
        
        fig, axs = plt.subplots(1,figsize=(8,8))
    
        plt.xlim([lbound[i],rbound[i]])
        plt.ylim([lbound[2*i+1],rbound[2*i+1]])
        
        plt.title("Dimensions %d and %d %s" % (2*i+1,2*i+2,note))
        plt.xlabel("Parameter number %d" % (2*i+1))
        plt.ylabel("Parameter number %d" % (2*i+2))
        
        targets = list(itertools.permutations(real_parameters))
        axs.scatter([t[0] for t in targets],[t[1] for t in targets],
                    marker='x',s=100, color='red')
        
        x1 = [particle[i] for particle in particles]
        x2 = [particle[i+1] for particle in particles]
        weights = [distribution[key]*200 for key in keys]
        axs.scatter(x1, x2, marker='o',s=weights)
        
    if dim%2!=0:
        # The last, unpaired dimension will be plotted alone with indexes as x
        #values.
        global first_plot_distribution
        if first_plot_distribution is True:
            print("> Dimension is odd, cannot combine parameters pairwise; one"
                  " will be plotted alone.[plot_distribution]")
        first_plot_distribution = False
        
        d = dim-1 
        keys = list(distribution.keys())
        particles = [np.frombuffer(key,dtype='float64')[d] for key in keys]
        targets = real_parameters
        
        fig, axs = plt.subplots(1,figsize=(8,8))
        plt.ylim([lbound[d],rbound[d]])
        plt.title("Dimension %d %s" % (d+1,note))
        plt.xlabel("Particle index (for visualization, not identification)")
        plt.ylabel("Parameter number %d" % (d+1))
        
        particle_enum = list(enumerate(particles))
        particle_indexes = [pair[0] for pair in particle_enum]
        particle_locations = [pair[1] for pair in particle_enum]
        
        weights = [distribution[key]*200 for key in keys]
        axs.scatter(particle_indexes,particle_locations,s=weights)
 
        [axs.axhline(y=target, color='r',linewidth=0.75,linestyle="dashed") 
         for target in targets] 
        
def generate_prior(distribution_type="uniform"):    
    '''
    Creates a uniform, random (and assymptotically uniform), or (assymptotically) 
    gaussian discrete distribution on some region.
    Note that for SMC to actually be considering a non-uniform prior, the code 
    should be changed to accomodate that (i.e. it affects the likelihood, log-
    likelihood and log-likelihood gradient in ways that aren't cancelled out
    by virtue of the effect being identical for all particles).
    
    Parameters
    ----------
    distribution_type: str, optional
        The class of distribution to be used (should be uniform, random or 
        gaussian/normal)
        
    Returns
    -------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        A dictionary representing the requested distribution (SMC 
        approximation).
    '''
    print("[Generated %s prior distribution.]" % (distribution_type))
    
    if distribution_type=="uniform":
        each = int(N_particles**(1/dim)) # Number of particles along each 
        #dimension.
        width = [rbound[i]-lbound[i] for i in range(dim)] # Parameter spans.
        
        # Create evenly spaced lists within the region defined for each 
        #parameter.
        particles = [np.arange(lbound[i]+width[i]/(each+1),rbound[i],
                       width[i]/(each+1)) for i in range(dim)]
        
        # Form list of all possible tuples (outer product of the lists for each 
        #parameter).
        particles = list(itertools.product(*particles))
        
    elif distribution_type=="random":
        particles = [[random.uniform(lbound[i],rbound[i]) 
                      for i in range(dim)] for j in range(N_particles)]
        
    elif distribution_type=="gaussian" or "normal":
        width = [rbound[i]-lbound[i] for i in range(dim)] # Parameter spans.
        particles = []
        while len(particles)<N_particles:
            new_particle = lbound-1 # Start with any set of invalid points.
            for i in range(dim):
                while new_particle[i]<lbound[i] or new_particle[i]>rbound[i]:
                    new_particle[i] = np.random.normal(width[i]/2,
                                                    scale=(width[i]/2)**2) 
                particles.append(new_particle)              
    else:  
        print("> `distribution_type` should be either uniform, random or "
              "gaussian. [generate_prior]")
        
    # Convert list to dictionary with keys as particles and uniform weights
    #(the distribution is characterized by particle density).
    prior = {}
    for particle in particles:
        particle = np.asarray(particle)
        # Use bit strings as keys because numpy arrays are not hashable.
        key = particle.tobytes()
        prior[key] = 1/N_particles
        
    return(prior)

def print_stats():
    '''
    Prints some relevant information relative to the run of the algorithm on
    the console.
    '''
    print("> n=%.2f^%d; N=%d; %dd sum of squared cosines" % 
                          (N_particles**(1/dim),dim,measurements,dim))
    
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

def sum_distributions(distributions):
    '''
    Sums a list of distributions, yielding the normalized sum of weights for 
    each point.
    
    Parameters
    ----------
    distributions: [dict]
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        A list of the dictionaries representing the distributions to be added 
        together (SMC approximations).
        
    Returns
    -------
    final_dist: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        A dictionary representing the resulting distribution (SMC 
        approximation).
    '''
    # Get a list of all unique keys.
    all_keys = set.union(*[set(d) for d in distributions])
    # The sum of the distributions is given by the normalized sum of weights
    #for each key. Absence means of course a weight of 0.
    final_dist = {key: np.sum([d.get(key, 0) for d in distributions])/
                  len(distributions) for key in all_keys}
    return final_dist

def offline_estimation(distribution, data, tempering_coefficients, 
                       threshold=None, plot_all=False):
    '''
    Estimates the vector of parameters by defining a set of experiments (times)
    , performing them, and updating a given prior distribution according to the
    outcomes (using Bayesian inference).
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The prior distribution (SMC approximation).
    data: [([float],int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    coef: float
        The sequence of tempering coefficients to be used in computing the 
        annealed likelihoods.
    threshold: float, optional
        The threshold effective sample size that should trigger a resampling 
        step when updating the distribution (Default is None, N_particles/2 
        will be used given the current value of the global variable 
        N_particles). 
    plot_all: bool, optional
        Whether to plot the particle positions at each step (Default is False).
        
    Returns
    -------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The final distribution (SMC approximation).
    '''
    
    if len(data)==0 or len(tempering_coefficients)==0:
        return
    
    if threshold is None:
        threshold = N_particles/2
        
    ans=""; resampled=False; counter=0; print("|0%",end="|")
    updates = len(tempering_coefficients)-1
    if updates < 10:
        progress_interval = 100/updates
    for i,coef in enumerate(tempering_coefficients[1:],start=1):
        if plot_all:
            if updates>10:
                while ans!="Y" and ans!="N":
                    ans = input("\n> This is going to print over 10 graphs. "\
                                "Are you sure you want that?"\
                                " [offline_estimation]\n(Y/N)\n")
            else:
                ans = "Y"
            if ans=="Y":
                info = "- step %d" % (i)
                info += " [resampled]" if resampled else ""
                plot_distribution(distribution,real_parameters,note=info)
            
        # Update the distribution: get the posterior of the current iteration, 
        #which is the prior for the next.
        distribution, resampled = bayes_update(data, distribution, coef, 
                tempering_coefficients[i-1], threshold, signal_resampling=True) 
        if resampled is None:
            print("> Process interrupted at iteration %d due to non-invertible"
                  " covariance matrix. [offline_estimation]" % i,end="")
            if plot_all:
                plot_distribution(distribution,real_parameters, 
                  note="(interrupted; importance sampled but no Markov moves)")
            break
        
        # Print up to 10 progress updates spaced evenly through the loop.
        if updates < 10:
            counter+=progress_interval
            print(round(counter),"%",sep="",end="|")
        elif (i%(updates/10)<1): 
            counter+=10
            print(counter,"%",sep="",end="|")
    print("") # For newline.
    return distribution

def main():
    global real_parameters, N_particles, measurements
    
    random_parameters = False
    if random_parameters:
        real_parameters = np.array([random.random() for d in range(dim)])
    else:
        real_parameters = np.array([0.8]) 
        #real_parameters = np.array([0.25,0.77]) 
        #real_parameters = np.array([0.25,0.77,0.40,0.52])
    
    measurements = 250
    steps = 5
    coefs = [i/steps for i in range(steps+1)] # These coefficients could be
    #chosen adaptively to keep the effective sample size near some target value
    #, but evenly spaced coefficients seem to work well for this problem and 
    #don't require extra Bayes updates. 

    t_max = 100
    ts = [t_max*random.random() for k in range(measurements)] 
    data=[(t,measure(t)) for t in ts]
    print("Offline estimation: random times <= %d" % t_max)
    print("Tempering coefficients: ",coefs)
    
    test_resampling, test_no_resampling = True, True
    
    if test_no_resampling: # Just for reference.
        N_particles = 12**dim
        prior = generate_prior(distribution_type="uniform")
        dist_no_resampling = offline_estimation(copy.deepcopy(prior),data,coefs,
                                                threshold=0,plot_all=False)
        plot_distribution(dist_no_resampling,real_parameters,
                          note="(no resampling)")
        print("No resampling test completed.")
    
    if test_resampling:
        global first_bayes_update, first_offline_estimation
        first_bayes_update, first_offline_estimation = True, True

        groups = 1 # The algorithm will be ran independently for `groups`
        #particle groups, on the same data. Their results will be joined 
        #together in the end.
        
        N_particles = 12**dim # For each group. Should be a power with integer
        #base and exponent `dim` so the particles can be neatly arranged into a
        #cubic latice for the prior (unless not using a uniform distribution).
        prior = generate_prior(distribution_type="uniform")
        
        final_dists = []
        for i in range(groups):
            print("~ Particle group %d (of %d) ~" % (i+1,groups))
            final_dists.append(offline_estimation(copy.deepcopy(prior),data,
                         coefs,threshold=float('inf'),plot_all=True))
            
        N_particles = N_particles*groups # To get the correct statistics. 
        final_dist = sum_distributions(final_dists)
        plot_distribution(final_dist,real_parameters)

    print_stats()
    
main()