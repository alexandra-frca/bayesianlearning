# -*- coding: utf-8 -*-
"""
Module with likelihood-related functions.
"""
from autograd import grad, numpy as np
import random
import global_vars as glob

first_measure = True

def init_likelihoods():
    '''
    Initializes constants pertaining to the module.
    '''
    global first_measure
    first_measure = True

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
    real_parameters, dim = glob.real_parameters, glob.dim
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
    p=np.sum(np.cos(particle*t/2)**2/glob.dim)
    return p 

def likelihood(data,particle):
    '''
    Provides an estimate for the likelihood  P(data|v,t) given a data record
    and a vector of parameters v (the particle).
    
    Parameters
    ----------
    data: [([float],int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    particle: [float]
        The set of parameters to be used for the likelihood.
        
    Returns
    -------
    p: float
        The estimated probability of obtaining the input outcome. 
    '''
    p = np.prod([simulate_1(particle,t) if (outcome==1)
                        else 1-simulate_1(particle,t) for (t,outcome) in data])
    return p 

def target_U(data,particle):
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
        
    Returns
    -------
    U: float
        The value of the "energy".
    '''
    U = -np.log(likelihood(data, particle)) 
    return (U)

def U_gradient(data,particle,autograd=False):
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
    autograd: bool, optional
        Whether to use automatic differenciation (Default is False).
        
    Returns
    -------
    DU: [float]
        The gradient of the "energy".
    '''
    dim = glob.dim
    if autograd:
        DU_f = grad(target_U,1)
        DU = np.array(DU_f(data,particle))
    else:
        DU = np.array(np.zeros(dim))
        for (t,outcome) in data:
            arg = particle*t/2
            L1 = np.sum(np.cos(arg)**2)/dim
            dL1 = -t/dim*np.cos(arg)*np.sin(arg)
            if outcome==1:
                DU -= dL1/L1
            if outcome==0:
                L0,dL0 = 1-L1,-dL1
                DU -= dL0/L0
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