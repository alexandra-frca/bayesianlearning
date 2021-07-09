# -*- coding: utf-8 -*-
"""
Hamiltonian learning implementation for estimating a frequency and a dephasing
factor, using offline Bayesian inference with tempered likelihood estimation.
The qubit is assumed to be initialized at state |1> for each iteration, and to
evolve under H = f*sigma_x/2, apart from the exponential decay resulting from 
the presence of decoherence. Estimation is performed for both the precession
frequency and the coherence time (or its inverse).
A sequential Monte Carlo approximation is used to represent the probability 
distributions, using random walk Metropolis-Hastings mutation steps (HMC has
to be adapted to tempered coefficients if used)
The evolution of the standard deviations with the steps is plotted, and the 
final  values of these quantities (in addition to the actual error) are 
printed.
"""

import sys, copy, random, pickle, matplotlib.pyplot as plt
from autograd import grad, numpy as np
np.seterr(all='warn')
dim=2
total_HMC, accepted_HMC = 0, 0
total_MH, accepted_MH = 0, 0

N_particles = 15**2 # Number of samples used to represent the probability
#distribution, using a sequential Monte Carlo approximation.

f_real, alpha_real = 0, 0 

left_boundaries = np.array([0,1/50])
right_boundaries = np.array([10,1/2])

def measure(t, particle=np.array([f_real,alpha_real])):
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
    r = np.random.binomial(1, 
                           p=(np.cos(2*np.pi*f_real*t/2)**2*np.exp(-alpha_real*t)+
                              (1-np.exp(-alpha_real*t))/2))
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
    test_f, test_alpha = particle
    p=np.cos(2*np.pi*test_f*t/2)**2*np.exp(-test_alpha*t)+\
        (1-np.exp(-test_alpha*t))/2
    return p 

def likelihood(data,particle, coef):
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
    # if np.size(data)==2: # Single datum case.
    #     t,outcome = data if len(data)==2 else data[0] # May be wrapped in array.
    #     p = simulate_1(particle,t) if outcome==1 else (1-simulate_1(particle,t))
    # else:
    #     p = np.product([likelihood(datum, particle) for datum in data])
    l = np.product([simulate_1(particle,t) if (outcome==1)
                        else 1-simulate_1(particle,t) for (t,outcome) in data])
    l = l**coef
    return l

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
    f, alpha = particle
    if autograd:
        DU_f = grad(target_U,1)
        DU = np.array(DU_f(data,particle))
    else:
        DU = np.array([0.,0.])
        for (t,outcome) in data:
            arg = 2*np.pi*f*t/2
            L1= np.cos(arg)**2*np.exp(-alpha*t)+(1-np.exp(-alpha*t))/2
            dL1df = -2*np.pi*t*np.exp(-alpha*t)*np.sin(arg)*np.cos(arg)
            dL1da = -t*np.exp(-alpha*t)*np.cos(arg)**2+t*np.exp(-alpha*t)/2
            if outcome==1:
                DL1 = np.array([dL1df/L1,dL1da/L1])
                DU -= DL1
            if outcome==0:
                L0 = 1-L1
                dL0df = -dL1df
                dL0da = -dL1da
                DL0 = np.array([dL0df/L0,dL0da/L0])
                DU -= DL0
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
    print("Autodiff: ", U_gradient(data,particle,autograd=True))
    print("Analytical: ",U_gradient(data,particle,autograd=False))

first_metropolis_hastings_step = True
def metropolis_hastings_step(data, particle, coef, S=np.identity(2),
                             factor=0.05,
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
          Sigma = "Sigma"
      else:
          Sigma = "I"
      print("MH:  Sigma=%s**0.5, factor=%.4f" % (Sigma,factor))
      first_metropolis_hastings_step = False

    global dim
    Sigma = factor*S**0.25 # Numpy normal receives standard deviation, not cov.
    # Start with any invalid point.
    new_particle = np.array([left_constraints[i]-1 for i in range(dim)]) 
    
    # Get a proposal that satisfies the constraints.
    while any([new_particle[i]<left_constraints[i] for i in range(dim)] + 
                  [new_particle[i]>right_constraints[i] for i in range(dim)]):
        new_particle = np.array([np.random.normal(particle[i], Sigma[i][i])
                                 for i in range(dim)])

    p = likelihood(data,new_particle,coef)/likelihood(data,particle,coef)
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
def hamiltonian_MC_step(data, particle,coef, 
                        M=np.identity(2), L=20, eta=0.01, threshold=1):
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
        
resampler_calls = 0
first_bayes_update = True
def bayes_update(data, distribution, coef, previous_coef, moves=1, threshold=1):
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
        The prior distribution (SMC approximation). 

    Returns
    ------- 
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle=[f,alpha]:=[frequency,decay factor] (as a bit string)
        The updated distribution (SMC approximation).
    resampled: bool
        Whether resampling has occurred.
    '''
    global first_bayes_update
    if first_bayes_update:
        print("Resampling threshold: %.2f; performing %d Markov move(s) per"
        " particle when resampling. [bayes_update]" % (threshold,moves))
        first_bayes_update = False

    global N_particles, resampler_calls
    acc_weight, acc_squared_weight = 0, 0
    
    for key in distribution:
        particle = np.frombuffer(key,dtype='float64')
        new_weight = likelihood(data,particle,coef-previous_coef)*distribution[key]
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
            for m in range(moves):
                particle = hamiltonian_MC_step(data,particle,coef,M=Cov_inv)
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

def plot_distribution(distribution, note=""):
    '''
    Plots a discrete distribution, by scattering points as circles with 
    diameters proportional to their weights. 
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The distribution to be plotted (SMC approximation).
    note: str, optional
        Some string to be appended to the graph title (Default is ""). 
    '''
    lbound, rbound = left_boundaries, right_boundaries
    keys = list(distribution.keys())
    particles = [np.frombuffer(key,dtype='float64') for key in keys]
    
    fig, axs = plt.subplots(1,figsize=(8,8))
    i=0

    plt.xlim([lbound[i],rbound[i]])
    plt.ylim([lbound[2*i+1],rbound[2*i+1]])
    
    plt.title("Dimensions %d and %d %s" % (2*i+1,2*i+2,note))
    plt.xlabel("Parameter number %d" % (2*i+1))
    plt.ylabel("Parameter number %d" % (2*i+2))
    
    x1 = [particle[i] for particle in particles]
    x2 = [particle[i+1] for particle in particles]
    weights = [distribution[key]*200 for key in keys]
    axs.scatter(x1, x2, marker='o',s=weights)

def show_results(off_runs,off_dicts,fs,alphas,parameters):
    '''
    Processes results and plots and prints statistical quantities of interest. 
    
    Parameters
    ----------
    off_runs: ([float],[float])
        A tuple containing:
        - A list of length-2 lists of means [mean_f,mean_alpha]);
        - A list of length-2 lists of standard deviations [std_f,std_alpha];
        Both by increasing iteration order.
    off_dicts: [dict]
        A list of final distributions respecting different runs.
    fs: [float]
        A list of final frequency means. Could be gotten from 'off_runs' or 
        'off_dicts', but this makes it easier.
    alphas: [float]
        A list of final dephasing factor means. Same as 'fs'.
    parameters: [float]
        The list of real parameters, for computing the mean squared errors. Will
        be an approximation if they're not exact (but based on IBM backend 
        properties) and the data is imported.
    '''
    #####
    '''
    The indexes in off_runs are, by order: 
        - Run number;
        - Desired quantity (0 for mean, 1 for stdev, 2 for cumulative_time)
        - Step number
        - Desired parameter (0 for frequency, 1 for alpha)
    '''
    runs = len(off_runs)
    steps = len(off_runs[0][0])-1

    '''
    # Get run with the median last step variance to print its mean squared error
    #instead of calculating all of them since we won't plot them.
    fstdevs = [s[steps-1][0] for m,s,t in off_runs]
    median = np.percentile(fstdevs, 50, interpolation='nearest')
    median_index = fstdevs.index(median)
    median_dist = off_dicts[median_index]
    mses = meansquarederror(median_dist,parameters[median_index])'''

    # Get run with the median last step frequency to plot.
    #fstdevs = [s[steps-1][0] for m,s,t in off_runs]
    median = np.nanpercentile(fs, 50, interpolation='nearest')
    median_index = fs.index(median)
    median_dist = off_dicts[median_index]
    plot_distribution(median_dist,note=" (run with median frequency)")
    # Same for decay factor.
    median = np.nanpercentile(alphas, 50, interpolation='nearest')
    median_index = alphas.index(median)
    median_dist = off_dicts[median_index]
    plot_distribution(median_dist,note=" (run with median α)")

    mses = meansquarederror(median_dist,parameters[median_index])
    all_mses = [meansquarederror(dist,parameters[i]) \
            for i,dist in enumerate(off_dicts)]
    median_mses = [np.median([mse[d] for mse in all_mses]) for d in range(dim)]

    off_errors = [[abs(off_runs[i][0][steps-1][p]-parameters[i][p]) 
                  for i in range(runs)] for p in range(2)]  

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
    
    median_f = np.median(fs)
    median_alpha = np.median(alphas)
    print("----------------")
    print("Median results: ")
    print("- f =        %.4f   ± %.4f" % (median_f,off_stdevs[0][steps]))
    T2 = 1/median_alpha
    print("- α = %.6f ± %.6f (T2* = %.2f ± %.2f)" % 
          (median_alpha, off_stdevs[1][steps], 
           T2, T2*off_stdevs[1][steps]/median_alpha))
    print("(𝛿T2 = T2 * 𝛿α/α)")
    print("- Variance:  %.6f; %.8f\n- MSE:       %.6f, %.8f\n"
    "- Deviation: %.4f; %.6f"       % (off_stdevs[0][steps]**2, 
                                      off_stdevs[1][steps]**2, 
                                      median_mses[0], median_mses[1],
                                      off_error[0],
                                      off_error[1]))
    
    fig, axs = plt.subplots(2,figsize=(8,8))
    fig.subplots_adjust(hspace=0.55)

    p=0
    x1 = np.array([i for i in range(steps+1)])
    oy1 = np.array([off_stdevs[p][i] for i in range(steps+1)])
    axs[0].set_ylabel(r'$\sigma$')
    axs[0].plot(x1, oy1, color='red', label='offline estimation')
    oq11 = np.array([off_stdevs_q1s[p][i] for i in range(steps+1)])
    oq31 = np.array([off_stdevs_q3s[p][i] for i in range(steps+1)])
    axs[0].fill_between(x1,oq11,oq31,alpha=0.1,color='red')
    axs[0].set_title('Frequency Estimation')
    axs[0].set_xlabel('Iteration number')
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    p=1
    oy1 = np.array([off_stdevs[p][i] for i in range(steps+1)])
    axs[1].set_ylabel(r'$\sigma$')
    axs[1].plot(x1, oy1, color='red', label='offline estimation')
    oq11 = np.array([off_stdevs_q1s[p][i] for i in range(steps+1)])
    oq31 = np.array([off_stdevs_q3s[p][i] for i in range(steps+1)])
    axs[1].fill_between(x1,oq11,oq31,alpha=0.1,color='red')
    axs[1].set_title('Coherence Factor Estimation')
    axs[1].set_xlabel('Iteration number')
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
def plot_grid(data, distribution, note=""):
    '''
    Plots a discrete distribution, by scattering points as circles with 
    diameters proportional to their weights. Also signalizes the target modes.
    A graph will be produced for each pair of dimensions.
    
    Parameters
    ----------
    data: [(float,int)]
        A vector of (evolution time, outcome) tuples.
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The distribution to be plotted (SMC approximation).
    note: str, optional
        Some string to be appended to the graph title (Default is ""). 
    '''
    lbound, rbound = left_boundaries, right_boundaries
    keys = list(distribution.keys())
    particles = [np.frombuffer(key,dtype='float64') for key in keys]
    
    fig, axs = plt.subplots(1,figsize=(8,8))
    i=0

    plt.xlim([lbound[i],rbound[i]])
    plt.ylim([lbound[2*i+1],rbound[2*i+1]])
    
    plt.title("Dimensions %d and %d %s" % (2*i+1,2*i+2,note))
    plt.xlabel("Parameter number %d" % (2*i+1))
    plt.ylabel("Parameter number %d" % (2*i+2))
    
    x1 = [particle[i] for particle in particles]
    x2 = [particle[i+1] for particle in particles]
    weights = [likelihood(data,particle)*500 \
               for particle in particles]
    axs.scatter(x1, x2, marker='o',s=weights)

def test_on_grid(data, side):
    '''
    Plots a grid of points with diameters proportional to their likelihoods 
    (given some data vector).
    
    Parameters
    ----------
    data: [(float,int)]
        A vector of (evolution time, outcome) tuples.
    side: int, optional
        The side of the 'grid' of particles to be considered.
    '''

    global f_real, alpha_real, N_particles
    #f_real, alpha_real = 1.83, 1/12
    N_particles = side**dim
    grid = uniform_prior()

    plot_grid(data, grid, note=" (grid likelihood plot)")
    
def print_stats(runs,steps):
    '''
    Prints some numbers related to the resampler and settings.
    
    Parameters
    ----------
    runs: int
        The total number of runs.
    steps: int
        The total number of iterations of each run.
    '''
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
    print("(n=%d; runs=%d; 2d, tempered estimation)" % (N_particles,runs))

def uniform_prior():
    '''
    Generates a flat prior distribution within the defined boundaries.
    
    Returns
    -------
    prior: dict
        , with (key,value):=(particle,importance weight) 
        , and particle=[f,alpha]:=[frequency,decay factor] (as a bit string)
        The prior distribution (SMC approximation).

    '''
    f_min, alpha_min = left_boundaries
    f_max, alpha_max = right_boundaries
    each = int(round(N_particles**(1/dim)))
    fs = np.linspace(f_min,f_max,each)
    alphas = np.linspace(alpha_min,alpha_max,each)
    prior = {}
    for f in fs:
        for alpha in alphas:
            particle = np.array([f,alpha])
            # Use bit strings as keys because numpy arrays are not hashable.
            key = particle.tobytes()
            prior[key] = 1/N_particles
    return prior

def offline_estimation(distribution, data, tempering_coefficients, plot=False):
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
    data: [(float,int)]
        A vector of (evolution time, outcome) tuples.
    plot: bool, optional
        Whether to plot the distribution a few times throughout (Default is 
        False).
        
    Returns
    -------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle=[f,alpha]:=[frequency,decay factor] (as a bit string)
        The posterior distribution (SMC approximation).
    means: [float]
        A list of the consecutive distribution means, including the prior's and
        the ones resulting from every intermediate step.
    stdevs: [float]
        A list of the consecutive distribution standard deviations, including 
        the prior's and the ones resulting from every intermediate step.
    _: str
        A placeholder for compatibility with how 'show_results()' was originally
        structured.
    ''' 

    current_mean, current_stdev = SMCparameters(distribution)
    means, stdevs = [], []
    means.append(current_mean)
    stdevs.append(current_stdev) 

    updates = len(tempering_coefficients)-1
    if updates < 10:
        progress_interval = 100/updates
    print("|0%",end="|"); counter = 0
    resampler_calls = 0
    for i,coef in enumerate(tempering_coefficients[1:],start=1):
        if plot and i%(updates/10)==0: # Generate 10 distribution plots.
            info = "- step %d" % (i)
            info += " (resampled %d time(s))" % resampler_calls
            resampler_calls = 0
            plot_distribution(distribution,note=info)

        distribution,resampled = bayes_update(data, distribution, coef,
                                              tempering_coefficients[i-1]) 
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
        plot_distribution(distribution,note=" (final distribution)")
    return distribution, (means, stdevs, _)

def get_data(upload=False, filename=None, every=1, steps=75, rep=1, rev=False,
             tmin=0.02, tmax=3.5, rand=False):
    '''
    Provides a data vector for the inference, either by loading it from a file 
    or by generating it.
    
    Parameters
    ----------
    upload: bool, optional
        Whether to load the data stored in a file (Default is False).
    filename: str, optional
        The name of the file from where the data should be loaded if 'upload' is
        True (Default is None).
    steps: int, optional
        The number of measurements to perform if upload is False (Default is 
        75).
    tmax: float, optional
        The maximum evolution time to be used if upload is False (Default is 
        3.5).
    rand: bool, optional
        Whether to choose the evolution times at random from [0,tmax[ if upload 
        is False (Default us False, the times will be spaced evenly instead).
        
    Returns
    -------
    data: [(float,int)]
        A vector of (evolution time, outcome) tuples.
    '''
    if upload:
        print("> Uploading data from file \'%s\'..." % filename)
        with open(filename, 'rb') as filehandle: 
            data = pickle.load(filehandle)
            # Overwrite step number and maximum time.
            steps, tmax = len(data), max([t for t,outcome in data])
            # Flip the outcomes because the code is structured oppositely to
            #the IBM experiments.
            data = [(t,outcome^1) for t,outcome in data]
            data = data[::every]
            if rev:
                data = data[::-1]
            steps, tmin, tmax = len(data), min([t for t,outcome in data]), \
                max([t for t,outcome in data])
    else:
        if rand:
            ts = [random.uniform(tmin,tmax) for i in range(steps)]
        else:
            ts = np.linspace(tmin,tmax,steps)
            ts = np.repeat(ts,rep)
            if rev:
                ts = ts[::-1]
            steps = len(ts)
        print(("> Using randomly generated " if rand else 
              "> Using evenly spaced ordered ") + "times.")
        data = [(t,measure(t)) for t in ts]
    print("Data vector:", data)
    print("> t∈[%.1f,%.1f[; steps = %d; f∈[%.2f,%.2f[; α∈[%.2f,%.2f[ (T2*∈[%.2f,%.2f[)" 
           % (tmin,tmax,steps,left_boundaries[0],right_boundaries[0],
              left_boundaries[1], right_boundaries[1],
              1/right_boundaries[1],1/left_boundaries[1]))
    return data,steps

def main():
    global f_real, alpha_real, N_particles, right_boundaries, dim
    f_max, alpha_max = right_boundaries
    f_real, alpha_real = 1.83, 1/15
    print("> f = %.2f, alpha = %.2f (T2* = %.2f)\n" % 
          (f_real,alpha_real,1/alpha_real))

    prior = uniform_prior()
    plot_distribution(prior,note=" (prior)")
    steps = 150
    coefs = [i/steps for i in range(steps+1)]
    #coefs = [0]+[(1/2)**((steps-k)) for k in range(1,steps+1)]

    print("Tempering coefficients: ", coefs)
    datasets = 10
    runs_each = 1
    print("> Will use %d datasets for %d runs each to compute the statistics." 
          % (datasets,runs_each))
    runs = datasets*runs_each
    off_runs = []
    parameters = [np.array([f_real,alpha_real]) for i in range(runs)]
    fs, alphas = [], []
    off_dicts = []
    try:
          for i in range(datasets):
              print(f"> Dataset {i}.")

              filename = 'guadalupe_ramsey_data[2.0,5[f=2_sched=75_nshots=2_' + str(i) + '.data'
              '''
              filename = 'ramsey_data[0.2,5[df=1.83_sched=75_nshots=1_' + str(i) + '.data' if i<5 \
                  else 'ramsey_data[0.2,5[df=1.83_sched=75_nshots=5_' + str(i%5) + '.data'
              every = 1 if i<5 else 5
              '''
              data,meas = get_data(filename=filename, every=1, upload=False, 
                                 steps=75, rep=2, rev=False, tmin=0.2, tmax=5, rand=False)

              for j in range(runs_each):
                  (dist,sequences) = offline_estimation(copy.deepcopy(prior),
                                                        data, coefs, plot=False)
                  off_runs.append(sequences)
                  off_dicts.append(dist)
                  f,alpha = sequences[0][steps]
                  fs.append(f)
                  alphas.append(alpha)
                  print("> Estimated f=%.2f and alpha=%.2f (T2=%.2f)."
                      % (f,alpha,1/alpha))
    except KeyboardInterrupt:
          err = sys.exc_info()[0]
          runs = len(fs)
          print("> Quit at run %d (%s)." % (runs,err))

    if runs!=0:
        show_results(off_runs,off_dicts,fs,alphas,parameters)
        print_stats(runs,steps)
        
main()