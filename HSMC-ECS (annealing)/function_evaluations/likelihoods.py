# -*- coding: utf-8 -*-
"""
Module with likelihood-related function evaluations.
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
    global first_measure
    if first_measure:
        print("Real parameters: ",real_parameters,end=".")
        print(" [function_evaluations.likelihoods.measure]")
        first_measure = False
    
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

def likelihood(data,particle,coef=1):
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
    coef: float, optional
        The tempering coefficient to be used (Default is 1).
        
    Returns
    -------
    l: float
        The annealed likelihood.
    '''
    
    l = np.product([simulate_1(particle,t) if (outcome==1)
                        else 1-simulate_1(particle,t) for (t,outcome) in data])
    l = l**coef
    return l

def loglikelihood(data,particle,coef=1):
    '''
    Evaluates the loglikelihood associated to the joint likelihood of some 
    vector  of data, given a set of parameters. 
    
    Parameters
    ----------
    data: [([float],int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    particle: [float]
        The set of dynamical parameters to be used for the likelihood.
    coef: float, optional
        The tempering coefficient to be used in computing the (annealed)
        loglikelihood (Default is 1).
        
    Returns
    -------
    l: float
        The value of the loglikelihood.
    '''
    l = np.log(np.product([simulate_1(particle,t) if (outcome==1)
                        else 1-simulate_1(particle,t) 
                        for (t,outcome) in data]))*coef
    return (l)

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
    U = -loglikelihood(data,particle,coef=coef)
    return (U)

def loggradient(data,particle,coef,autograd=False,ind=None):
    '''
    Evaluates the gradient of the target loglikelihood.
    
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
        Whether to use automatic differentiation (Default is False).
    ind: int, optional
        If this isn't null, the `ind`th element of the resulting vector will
        be returned instead of the whole array (Default is None).
        This is meant for testing purposes, to allow for the use the gradient  
        calculator for testing higher (namely second) order differentiation 
        (the autograd library doesn't support full Jacobians but is indirectly 
        suitable for diagonal ones, as is the case since the value of the 
        gradient along each dimension depends only on that dimension's 
        parameter). 
        
    Returns
    -------
    lg: [float]
        The gradient of the loglikelihood.
    '''
    dim = glob.dim
    if autograd:
        lg_f = grad(loglikelihood,1)
        lg = np.array(lg_f(data,particle,coef))
    else:
        lg = np.array(np.zeros(dim))
        for (t,outcome) in data:
            arg = particle*t/2
            L1 = np.sum(np.cos(arg)**2)/dim
            dL1 = -t/dim*np.cos(arg)*np.sin(arg)
            if outcome==1:
                lg += coef*dL1/L1
            if outcome==0:
                L0,dL0 = 1-L1,-dL1
                lg += coef*dL0/L0  
    if ind is not None:
        return lg[ind]
    return(lg)

def U_gradient(data,particle,coef):
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
        
    Returns
    -------
    DU: [float]
        The gradient of the "energy".
    '''
    DU = -loggradient(data,particle,coef)
    return(DU)

def sec_order_loggradient(data,particle,coef,autograd=False):
    '''
    Evaluates the second order gradient of the target loglikelihood.
    This is the diagonal of the Hessian matrix, and comprises all the
    information contained by it since these are the only possible non-zero 
    entries (the target likelihood is a sum of single-parameter terms).
    
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
        Whether to use automatic differentiation (Default is False).
        
    Returns
    -------
    lg: [float]
        The gradient of the loglikelihood.
    '''
    dim = glob.dim
    if autograd:
        D2l_f = grad(loggradient,1)
        D2l = np.array([D2l_f(data,particle,coef,ind=d)[d] 
                        for d in range(dim)])
    else:
        D2l = np.array(np.zeros(dim))
        for (t,outcome) in data:
            arg = particle*t/2
            L1 = np.sum(np.cos(arg)**2)/dim
            dL1 = -t/dim*np.cos(arg)*np.sin(arg)
            ddL1 = -t**2/(2*dim)*(-np.sin(arg)**2+np.cos(arg)**2)
            if outcome==1:
                D2l += coef*(ddL1*L1-dL1**2)/L1**2 
            if outcome==0:
                L0,dL0,ddL0 = 1-L1,-dL1,-ddL1
                D2l += coef*(ddL0*L0-dL0**2)/L0**2 
                     
    return(D2l)

def test_differentiation():
    '''
    Tests the analytical differentiation in `U_gradient` by comparing its
    numerical result against the one computed by automatic differentiation.
    Both are evaluated for the same representative set of inputs and printed on 
    the console.
    '''
    data = np.array([(random.random(),1),(random.random(),0)])
    particle = np.array([random.random() for d in range(glob.dim)])
    coef = random.random()
    print("* Gradient evaluation (1st order):")
    print("Autodiff: ", loggradient(data,particle,coef,autograd=True))
    print("Analytical: ",loggradient(data,particle,coef,autograd=False))
    print("* Gradient evaluation (2nd order):")
    print("Autodiff: ", sec_order_loggradient(data,particle,coef,
                                              autograd=True))
    print("Analytical: ",sec_order_loggradient(data,particle,coef,
                                               autograd=False))
    print("[test_differentiation]")
    