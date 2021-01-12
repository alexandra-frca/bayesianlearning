# -*- coding: utf-8 -*-
"""

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
    global phi_real
    r = random.random()
    p = np.random.binomial(1, 
                           p=(1-np.cos(M*(phi_real+theta)))/2)
    if (r<p):
        return 1
    return 0

def simulate_1(M, theta, phi):
    '''
    Provides an estimate for the likelihood  P(D=1|test_f,t) of an x-spin 
    measurement at time t yielding result |+>, given a test parameter for the 
    fixed form Hamiltonian. 
    This estimate is computed as the fraction of times a simulated system
    evolution up to time t yields said result upon measurement.
    
    Parameters
    ----------
    test_f: float
        The test precession frequency.
    t: float
        The evolution time between the initialization and the projection.
        
    Returns
    -------
    p: float
        The estimated probability of finding the particle at state |1>.
    '''
    p=(1-np.cos(M*(phi+theta)))/2
    return p 

def likelihood(data, test_phi):
    if np.size(data)==3:
        M, theta, outcome = data
        p = simulate_1(M,theta,test_phi)*(outcome==1)+\
            (1-simulate_1(M,theta,test_phi))*(outcome==0) 
    else:
        p = np.product([likelihood(datum, test_phi) for datum in data])
    return p 
    

def loglikelihood(data, test_phi):
    '''
    Provides an estimate for the likelihood  P(D|test_f,t) of an x-spin 
    measurement at time t yielding a given result, given a test parameter for  
    the fixed form Hamiltonian. 
    
    Parameters
    ----------
    outcome: int
        The result of the measurement (with |+> mapped to 1, and |-> to 0).
    test_f: float
        The test precession frequency.
    t: float
        The evolution time between the initialization and the projection.
        
    Returns
    -------
    p: float
        The estimated probability of obtaining the input outcome. 
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
    outcome: int
        The result of the measurement (with |+> mapped to 1, and |-> to 0).
    t: float
        The time at which the current iteration's measurement was performed.
        This is relevant because the target function and its derivative are 
        time-dependent.
    particle: float
        The particle to undergo a mutation step.
    s: float, optional
        The standard deviation to be multiplied by a factor and then used as 
        standard deviation for the normal distribution used for the proposal 
        (Default is 1).
    factor: float, optional
        The factor 's' should be be multiplied by to get the standard deviation
        of the the normal distribution used for the proposal (Default is 0.05).
        
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
    outcome: int
        The result of the measurement (with |+> mapped to 1, and |-> to 0).
    test_f: float
        The frequency to be used for the likelihood.
    t: float
        The evolution time between the initialization and the projection.
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

def simulate_dynamics(data, initial_momentum, initial_particle, m, L, eta,
                      factor=1):    
    '''
    Simulates Hamiltonian dynamics for a given particle, using leapfrog 
    integration.
    
    Parameters
    ----------
    outcome: int
        The result of the measurement (with |+> mapped to 1, and |-> to 0).
    t: float
        The time at which the current iteration's measurement was performed.
        This is relevant because the target function and its derivative are 
        time-dependent.
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
        if (l != L-1):
            DU = U_gradient(data,new_particle)
            new_particle = new_particle - eta*DU
    new_momentum = new_momentum - 0.5*eta*DU

    p = np.exp(-loglikelihood(data,initial_particle)-\
               (-loglikelihood(data,new_particle))+\
                   initial_momentum**2/(2*m)-new_momentum**2/(2*m))
    new_particle = new_particle%(2*np.pi)
    return new_particle, p

# Seems to work decent for: point, m=0.5, L=10, eta=10**-3,
# or m=0.05, L=10, eta=5*10**-4
# or m=0.05, L=20, eta=5*10**-3
# ...
# always with: threshold=0.01, sigma_MH=0.01 (or 0.1,...)
# And different MH/HMC ratios
first = True
def hamiltonian_MC_step(data, point, m=0.05, L=10, eta=5*10**-4,
                        threshold=0.01):
    '''
    Performs a Hamiltonian Monte Carlo mutation on a given particle.
    
    Parameters
    ----------
    t: float
        The time at which the current iteration's measurement was performed.
        This is relevant because the target function and its derivative are 
        time-dependent.
    particle: float
        The frequency particle to undergo a mutation step.
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
    particle: float
        The mutated frequency particle.
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
    '''
    counter=0
    while (p < threshold) and (counter<3):
        new_point, p = simulate_dynamics(data,initial_momentum,point,m,L,eta,
                                         factor=10**-(counter+1))
        counter += 1
        
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
    '''
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

    
def plot_likelihood(data, points=None, point_types=None):
    fig, axs = plt.subplots(1,figsize=(15,10))
    axs.set_title("Phase Estimation with MCMC (HMC/MH)",pad=20,fontsize=18)
    axs.set_ylabel("Likelihood",fontsize=14)
    axs.set_xlabel("Phase",fontsize=14)
    axs.tick_params(axis='y', colors="blue")
    axs.yaxis.label.set_color('blue')
    xx = np.arange(0.1,2*np.pi,0.1)
    yy = [likelihood(data,x) for x in xx]
    axs.plot(xx,yy,linewidth=1.75,alpha=0.5)
    global phi_real
    axs.axvline(phi_real,linestyle='--',color='gray',linewidth='1')
    
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

def offline_phase_estimation(measurements,steps):
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
    
    plot_likelihood(data,points=trajectory, point_types=proposals)

def adaptive_phase_estimation(measurements,steps):
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
    #return(mean,stdev)


def main():
    global phi_real,steps, measurements
    offline_phase_estimation(measurements,steps)
    #adaptive_phase_estimation(measurements,steps)
    
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