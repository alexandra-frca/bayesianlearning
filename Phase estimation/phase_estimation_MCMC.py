
# -*- coding: utf-8 -*-
"""
Estimates a phase by using Markov Chain Monte Carlo (with Hamiltonian Monte  
Carlo and Metropolis-Hastings transitions) to sample from a product of 
likelihoods.
"""

import random, matplotlib.pyplot as plt
from autograd import grad, numpy as np

phi_real = 0.5
measurements = 100
steps = 100

total_HMC, accepted_HMC = 0, 0
total_MH, accepted_MH = 0, 0

def measure(M, theta):
    '''
    Simulates a measurement on the phase estimation circuit; the probability 
    of 0 is P(0|phi;theta,M)=(1+cos(M*(phi+theta)))/2 and that of 1 is its 
    complement.
    
    Parameters
    ----------
    M: int
        The number of times the operator whose eigenvalue is to be determined is 
        applied (also used as a parameter for the rotation).
    theta: float
        A parameter for the rotation gate to be used in the circuit.
        
    Returns
    -------
    The obtained result (0 or 1).
    '''
    global phi_real
    r = np.random.binomial(1, 
                           p=(1-np.cos(M*(phi_real+theta)))/2)
    return r

def simulate_1(M, theta, phi):
    '''
    Provides an estimate for the likelihood  P(D=1|phi;theta,M) of a measurement 
    on the phase estimation circuit yielding result 1, given the experiment 
    controls and a phase (the test parameter).
    
    Parameters
    ----------
    M: int
        The number of times the operator whose eigenvalue is to be determined is 
        applied (also used as a parameter for the rotation).
    theta: float
        A parameter for the rotation gate to be used in the circuit.
    phi: float
        A phase (the test parameter).
    
    Returns
    -------
    p: float
        The estimated probability of finding the particle at state 1.
    '''
    p=(1-np.cos(M*(phi+theta)))/2
    return p 

def likelihood(data, test_phi):
    '''
    Provides an estimate for the likelihood  P(D|phi;theta,M) given a vector of 
    data (experimental outcomes and controls) and a test parameter (a phase).
    
    Parameters
    ----------
    data: [(int,float,int)]
        A vector of experimental results and controls, each datum being of the 
        form (M,theta,outcome), where 'M' and 'theta' are the controls used for 
        each experiment and 'outcome' is its result.
    test_phi: float
        A phase (the test parameter).
        
    Returns
    -------
    p: float
        The estimated probability of obtaining the data given the parameter phi. 
    '''
    if np.size(data)==3:
        M, theta, outcome = data
        p = simulate_1(M,theta,test_phi)*(outcome==1)+\
            (1-simulate_1(M,theta,test_phi))*(outcome==0) 
    else:
        p = np.prod([likelihood(datum, test_phi) for datum in data])
    return p 
    

def loglikelihood(data, test_phi):
    '''
    Provides an estimate for the log-likelihood  log(P(D|phi;theta,M)) given a   
    vector of data (experimental outcomes and controls) and a test parameter 
    (a phase).
    
    Parameters
    ----------
    data: [(int,float,int)]
        A vector of experimental results and controls, each datum being of the 
        form (M,theta,outcome), where 'M' and 'theta' are the controls used for 
        each experiment and 'outcome' is its result.
    test_phi: float
        A phase (the test parameter).
        
    Returns
    -------
    p: float
        The logarithm of the estimated probability of obtaining the data given 
        the parameter phi. 
    '''
    p = np.sum([np.log(likelihood(datum, test_phi)) for datum in data\
                if likelihood(datum, test_phi)!=0])
    return p 

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
def metropolis_hastings_step(data, particle, sigma=0.01): 
    '''
    Performs a Metropolis-Hastings mutation on a given particle, using a 
    gaussian function for the proposals.
    
    Parameters
    ----------
    data: [(int,float,int)]
        A vector of experimental results and controls, each datum being of the 
        form (M,theta,outcome), where 'M' and 'theta' are the controls used for 
        each experiment and 'outcome' is its result.
    particle: float
        The particle to undergo a mutation step.
    sigma: float, optional
        The standard deviation to be used in the normal distribution to generate
        proposals (Default is 0.01).
        
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

    new_particle = np.random.normal(particle, sigma)%(2*np.pi)
        
    # Compute the probabilities of transition for the acceptance probability.
    divisor = likelihood(data,particle)*gaussian(new_particle,particle,sigma)
    if divisor==0:
        p = 1
    else:
        p = likelihood(data,new_particle)*gaussian(particle,new_particle,
                                                        sigma)/divisor 
    return new_particle,p

def U_gradient(data,test_phi,autograd=False):
    '''
    Evaluates the derivative of the target "energy" associated to the likelihood 
    at a time t seen as a probability density, given a frequency for the fixed 
    form Hamiltonian. 
    
    Parameters
    ----------
    data: [(int,float,int)]
        A vector of experimental results and controls, each datum being of the 
        form (M,theta,outcome), where 'M' and 'theta' are the controls used for 
        each experiment and 'outcome' is its result.
    test_phi: float
        A phase (the test parameter).
    autograd: bool, optional
        Whether to use automatic differenciation (Default is False).
        
    Returns
    -------
    DU: float
        The derivative of the "energy".
    '''

    '''
    The cosine must be different from 1 if the outcome is 1, and from -1 if the
    outcome is 0 (so that the divisor is non-zero when evaluating the            
    derivative).
    '''
    usable_data = [(M,theta,outcome) for (M,theta,outcome) in data 
            if np.cos(M*(theta+test_phi))!=(-1)**(outcome^1)]

    if autograd: 
        minus_DU_f = grad(loglikelihood,1)
        DU = -minus_DU_f(usable_data,float(test_phi))
    else:
        DU = np.sum([(-M*np.sin(M*(theta+test_phi))\
                               /(1-np.cos(M*(theta+test_phi)))) if outcome==1
                 else (M*np.sin(M*(theta+test_phi))\
                               /(1+np.cos(M*(theta+test_phi))))
                 for (M,theta,outcome) in usable_data])
    return(DU)

def simulate_dynamics(data, initial_momentum, initial_particle, m, L, eta):    
    '''
    Simulates Hamiltonian dynamics for a given particle, using leapfrog 
    integration.
    
    Parameters
    ----------
    data: [(int,float,int)]
        A vector of experimental results and controls, each datum being of the 
        form (M,theta,outcome), where 'M' and 'theta' are the controls used for 
        each experiment and 'outcome' is its result.
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
        DU = U_gradient(data,new_particle)
        if (l != L-1):
            new_particle = new_particle - eta*DU
    new_momentum = new_momentum - 0.5*eta*DU

    p = np.exp(-loglikelihood(data,initial_particle)-\
               (-loglikelihood(data,new_particle))+\
                   initial_momentum**2/(2*m)-new_momentum**2/(2*m))
    new_particle = new_particle%(2*np.pi)
    return new_particle, p

# Seems to work well for: point, m=0.5, L=10, eta=10**-3,
# or m=0.05, L=10, eta=5*10**-4
# or m=0.05, L=20, eta=5*10**-3
# ...
# always with: threshold=0.01
first = True
def hamiltonian_MC_step(data, point, m=0.5, L=10, eta=10**-3,
                        threshold=0.01):
    '''
    Performs a Hamiltonian Monte Carlo mutation on a given particle.
    
    Parameters
    ----------
    data: [(int,float,int)]
        A vector of experimental results and controls, each datum being of the 
        form (M,theta,outcome), where 'M' and 'theta' are the controls used for 
        each experiment and 'outcome' is its result.
    point: float
        The point in parameter space (i.e. a phase) to undergo a transition.
    m: float, optional
        The mass to be used when simulating the Hamiltonian dynamics (a HMC 
        tuning parameter) (Default is 0.05).
    L: int, optional
        The amount of integration steps to be used when simulating the 
        Hamiltonian dynamics (a HMC tuning parameter) (Default is 10).
    eta: float, optional
        The integration stepsize to be used when simulating the Hamiltonian 
        dynamics (a HMC tuning parameter) (Default is 5*exp(-4)).
    threshold: float, optional
        The highest HMC acceptance rate that should trigger a Metropolis-
        -Hastings mutation step (as an alternative to a  HMC mutation step) 
        (Default is 0.01). 
        
    Returns
    -------
    point: float
        The point after the suggested transition (i.e. a phase parameter).
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
    if (p < threshold):
        proposal = "RWM"
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

def plot_likelihood_new(data, points, point_types=None,plot_gradient=False):
    '''
    new: put the step number axis on the left for clarity and increase tick 
    number sizes. 
    '''
    FONTSIZE = 40
    SMALLERSIZE = 30

    fig, axs = plt.subplots(1,figsize=(15,10))
    #axs.set_title("MCMC using HMC and MH proposals",pad=20,fontsize=18)
    axs.set_ylabel("Step number",fontsize=FONTSIZE)
    axs.set_xlabel("Frequency",fontsize=FONTSIZE)
    
    xx = np.arange(0.1,2*np.pi,0.01)
    yy = [likelihood(data,x) for x in xx]
    
        
    unique_types = ["Starting point","HMC","RWM"]
    for unique_type in unique_types:
        xi = [points[i] for i in range(len(points)) 
                if point_types[i]==unique_type]
        yi = [i for i in range(len(points)) if point_types[i]==unique_type]
        color = "red" if unique_type == "HMC" \
            else "black" if unique_type == "RWM" else "blue"
        marker_type = "x" if unique_type == "Starting point" else "."
        axs.scatter(xi, yi, marker=marker_type,s=500, c=color,
                        label=unique_type)
    
    
    axs2 = axs.twinx()
    axs2.yaxis.label.set_color('blue')
    
    axs2.set_ylabel("Likelihood",fontsize=FONTSIZE)
    axs2.plot(xx,yy,linewidth=5,alpha=0.5)

    plt.xticks(fontsize=SMALLERSIZE)
    axs.tick_params(axis='both', which='major', labelsize=SMALLERSIZE, length=15, width=2.5) 
    axs2.tick_params(colors="blue", axis='both', which='major', labelsize=SMALLERSIZE, length=15, width=2.5) 
    axs.legend(loc='best',fontsize=SMALLERSIZE)
    axs2.yaxis.get_offset_text().set_fontsize(SMALLERSIZE)

    plt.savefig('figs/ipe_mcmc.png', bbox_inches='tight')
    
    
def plot_likelihood(data, points=None, point_types=None):
    '''
    Plots - on the interval [0,2*pi[ - the likelihood function corresponding to 
    the given data (which is the product of the individual likelihoods of each 
    datum), as well as (optionally) a set of points (as an overposed scatter 
    plot).
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
    '''
    fig, axs = plt.subplots(1,figsize=(15,10))
    #axs.set_title("Phase Estimation with MCMC (HMC/MH)",pad=20,fontsize=18)
    axs.set_ylabel("Likelihood",fontsize=20)
    axs.set_xlabel("Phase (radians)",fontsize=20)
    axs.tick_params(axis='y', colors="blue")
    axs.yaxis.label.set_color('blue')
    xx = np.arange(0.1,2*np.pi,0.1)
    yy = [likelihood(data,x) for x in xx]
    axs.plot(xx,yy,linewidth=1.75,alpha=0.5)
    global phi_real
    axs.axvline(phi_real,linestyle='--',color='gray',linewidth='1')
    
    if points is not None:
        axs2 = axs.twinx()
        axs2.set_ylabel("Step number",fontsize=20)
        
        unique_types = ["Starting point","HMC","RWM"]
        for unique_type in unique_types:
            xi = [points[i] for i in range(len(points)) 
                 if point_types[i]==unique_type]
            yi = [i for i in range(len(points)) if point_types[i]==unique_type]
            color = "red" if unique_type == "HMC" \
                else "black" if unique_type == "RWM" else "blue"
            marker_type = "x" if unique_type == "Starting point" else "."
            axs2.scatter(xi, yi, marker=marker_type,s=300, c=color,
                         label=unique_type)
        axs2.legend(loc='upper right',fontsize=20)

def offline_phase_estimation(measurements,steps):
    '''
    Performs phase estimation using a pre-defined set of experimental controls.
    
    Parameters
    ----------
    measurements: int
        The total number of measurements to be performed.
    steps: int
        The total number of steps for which to evolve the Markov chain.
    '''
    global phi_real, N_samples
    data = []
    for i in range(round(measurements**0.5)):
        M = i+1
        for j in range(round(measurements**0.5)):
            theta = j*np.pi/5
            outcome = measure(M, theta)
            data.append((M,theta,outcome))

    point = np.pi # Start the chain at the middle.    
    trajectory, proposals = [point], ["Starting point"]    
    for j in range(steps):
        point,proposal = hamiltonian_MC_step(data,point)
        trajectory.append(point)
        proposals.append(proposal)
        #print(point)
    
    plot_likelihood_new(data,points=trajectory, point_types=proposals)

def adaptive_phase_estimation(measurements,steps):
    '''
    Performs phase estimation, using the data from experiments for which the 
    controls are chosen adaptively.
    
    Parameters
    ----------
    measurements: int
        The total number of measurements to be performed.
    steps: int
        The number of steps for which to evolve the Markov chain for each added
        measurement. The total number of steps will be measurements*steps.
        The Markov chain will undergo n='steps' transitions on each of the 
        cumulative products of likelihoods, meaning on the ith iteration of the 
        outer loop it will be sampling according to the probability density 
        arising from the joint data of the first i measurements.
    '''
    global phi_real
    data = []
    M,theta = 1,random.random()*2*np.pi # Arbitrary first measurement.
    outcome = measure(M, theta)
    data.append((M,theta,outcome))
    
    point = np.pi # Start the chain at the middle.
    trajectory, proposals = [point], ["Starting point"]
    sample_index = random.randint(0,steps-1)
    for i in range(measurements-1):
        acc_sum, acc_squares = 0,0
        shifted_sum, shifted_squares = 0,0
        
        for j in range(steps):
            point,proposal = hamiltonian_MC_step(data,point)
            trajectory.append(point)
            #print(point,proposal)
            proposals.append(proposal)
            if j==sample_index:
                sample = point
            # Try to take lag samples/discard a burn-in?
            acc_sum += point
            acc_squares += point**2
            shifted_phi = (point+np.pi)%(2*np.pi)
            shifted_sum += shifted_phi
            shifted_squares += shifted_phi**2
            
        stdev = np.sqrt(abs(acc_squares-np.power(acc_sum,2)/steps)\
                        /(steps-1))
        if stdev==0:
            break
        M,theta = np.ceil(1.25/stdev),sample 
        outcome = measure(M, theta)
        data.append((M,theta,outcome))
        
    plot_likelihood(data,points=trajectory, point_types=proposals) 

def main():
    global phi_real,steps, measurements
    steps=30
    offline_phase_estimation(measurements,steps)
    #adaptive_phase_estimation(measurements,steps)
    
    print("> %d measurements, %d steps, MCMC" % (measurements,steps))
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