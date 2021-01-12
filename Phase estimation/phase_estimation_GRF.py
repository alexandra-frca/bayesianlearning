# -*- coding: utf-8 -*-
"""
Estimates a phase by rejection filtering phase estimation, assuming a gaussian
model.
Based on "Efficient Bayesian Phase Estimation"
[https://arxiv.org/abs/1508.00869]
"""

import random, numpy as np, matplotlib.pyplot as plt

N_samples = 100
phi_real = 0.5

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

def adaptive_phase_estimation(measurements):
    global phi_real, N_samples
    M,theta = 1,random.random()*2*np.pi # Arbitrary first measurement.
    outcome = measure(M, theta)
    data = [(M,theta,outcome)]
    accepted = 0
    acc_sum, acc_squares = 0,0
    shifted_sum, shifted_squares = 0,0
    for j in range(N_samples):
        test_phi = random.random()*2*np.pi # Flat prior.
        likelihood_1 = simulate_1(M,theta,test_phi)
        likelihood = likelihood_1 if outcome==1 else (1-likelihood_1)
        if random.random() < likelihood:
            accepted += 1
            accepted_phi = test_phi%(2*np.pi)
            acc_sum += accepted_phi
            acc_squares += accepted_phi**2
            shifted_phi = (test_phi+np.pi)%(2*np.pi)
            shifted_sum += shifted_phi
            shifted_squares += shifted_phi**2
        
    mean = acc_sum/accepted
    trajectory = [mean]
    stdev = np.sqrt(abs(acc_squares-np.power(acc_sum,2)/accepted)/(accepted-1))
    shifted_stdev = np.sqrt(abs(shifted_squares-\
                                np.power(shifted_sum,2)/accepted)/(accepted-1))
    if (stdev > shifted_stdev):
        stdev = shifted_stdev
        shifted_mean = shifted_sum/accepted
        mean = (shifted_mean-np.pi)%(2*np.pi)
    
    for i in range(measurements):
        M,theta = np.ceil(1.25/stdev),np.random.normal(mean,scale=stdev)
        outcome = measure(M, theta)
        data.append((M,theta,outcome))
        accepted = 0
        acc_sum, acc_squares = 0,0
        shifted_sum, shifted_squares = 0,0
        for j in range(N_samples):
            test_phi = np.random.normal(mean,scale=stdev) # Gaussian prior.
            likelihood_1 = simulate_1(M,theta,test_phi)
            likelihood = likelihood_1 if outcome==1 else (1-likelihood_1)
            if random.random() < likelihood:
                accepted += 1
                accepted_phi = test_phi%(2*np.pi)
                acc_sum += accepted_phi
                acc_squares += accepted_phi**2
                '''
                We will need to compute the standard deviation and possibly
                mean for the pi-shifted distribution, to diagnose close to 
                0 mod 2pi frequencies that would be misrepresented by a 
                gaussian on the interval [0,2pi[ due to being split at the
                boundaries (with the right side to the right of 0 and the left
                side to the left of 2pi), yielding a close to pi mean and a
                larger than should be variance.
                Assuming sharp enough  distributions, these ~0 frequencies will        
                be the only ones suffering from this issue, and shifting them 
                by pi should allow us to compute the real standard deviation 
                and mean (by shifting the one obtained back by pi). 
                This correction should be triggered by the pi-shifted variance 
                being smaller than the original one; as such, these should both
                be computed at every step for screening purposes.
                In alternative, a wrapped normal distribution (on the unit 
                circle) could be used.
                '''
                shifted_phi = (test_phi+np.pi)%(2*np.pi)
                shifted_sum += shifted_phi
                shifted_squares += shifted_phi**2
                
        mean = acc_sum/accepted
        stdev = np.sqrt(abs(acc_squares-np.power(acc_sum,2)/accepted)\
                        /(accepted-1))
        shifted_stdev = np.sqrt(abs(shifted_squares-\
                                    np.power(shifted_sum,2)/accepted)\
                                /(accepted-1))
        if (stdev > shifted_stdev):
            stdev = shifted_stdev
            shifted_mean = shifted_sum/accepted
            mean = (shifted_mean-np.pi)%(2*np.pi)
            
        trajectory.append(mean)
        if stdev==0:
            break
        
    plot_likelihood(data,points=trajectory)
    return(mean,stdev)

def plot_likelihood(data, points=None):
    fig, axs = plt.subplots(1,figsize=(15,10))
    axs.set_title("Phase Estimation (gaussians)",pad=20,fontsize=18)
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
        axs2.set_ylabel("Iteration number",fontsize=14)
        y = [i for i in range(len(points))]
        axs2.scatter(points, y, marker=".",s=10,c="black",label="means")            

def main():
    global phi_real
    runs = 1 # Don't use many runs while plotting, will open lots of graphs
    means, stdevs = [], []
    for i in range(runs):
        mean, stdev = adaptive_phase_estimation(100)
        means.append(mean)
        stdevs.append(stdev)
    median = np.percentile(means,50,interpolation='nearest')
    median_stdev = stdevs[means.index(median)]
    print("Mean: ", mean,"\nStandard deviation: ", stdev)
    print("Median over %d runs; phi_real=%f" % (runs,phi_real))
    
main()