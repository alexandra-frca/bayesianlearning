# -*- coding: utf-8 -*-
"""
Module with estimator-related function evaluations.

Computes a collection of approximations and estimators; most relevantly, 
assymptotically unbiased estimators for the likelihood and loglikelihood.

The approximations serve as control variates to reduce the variance induced by 
subsampling. These variates are valid only for unimodal distributions, because 
they are Taylor-series based and evaluated at some center point (which should 
approximate the posterior location).

This means that for the target function (a multi-parameter sum of squared 
cosines), we must have either that dim=1 (single parameter case) or that the 
parameter takes the same value along each dimension (which must be built into
the model). 

If that is not the case, another choice of control variates must be used (or 
none at all, leaving only the subsampled quantities).
"""
from autograd import grad, numpy as np
import random
import global_vars as glob
from function_evaluations.likelihoods import likelihood, loglikelihood, \
    loggradient, sec_order_loggradient, test_differentiation, measure

def Taylor_coefs(data,center,quadratic=True):
    '''
    Computes the first 2 to 3 Taylor series coefficients for the loglikelihood.
    
    Parameters
    ----------
    data: [([float],int)]
        The vector of data results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    center: [float]
        The center for the Taylor expansion, i.e. the coordinates of the point 
        at which the 0 to 2nd order gradients should be evaluated.
    quadratic: bool, optional
        Whether to compute the coefficients up to second degree; if set to 
        False, only the 1st and 2nd order coefficients will be returned 
        (Default is True).
        
    Returns
    -------
    A: float
        The 0th order Taylor coefficient (loglikelihood at the center).
    B: float
        The 1st order Taylor coefficient (loggradient at the center).
    C: float
        The 2nd order Taylor coefficient (second order loggradient at the 
        center).
    '''
    A = np.log(likelihood(data,center))
    B = loggradient(data,center,1) 
    if not quadratic:
        return A,B
    C = sec_order_loggradient(data, center,1)
    return A,B,C

def approx_loglikelihood(particle,coef,mean,Tcoefs,quadratic=True):
    '''
    Provides an approximation to the (tempered) loglikelihood (Taylor series 
    around a point).
    
    Parameters
    ----------
    particle: [float]
        The set of parameters at which the loglikelihood is to be evaluated.
    coef: float
        The tempering coefficient for the likelihood.
    mean: [float]
        The mean particle, which defines the point around which the Taylor 
        expansion will be evaluated.
    Tcoefs: (float,float,float) 
        The 0-th to 2nd order Taylor coefficients. The last tuple item may be a
        dummy value if quadratic=False.
    quadratic: bool, optional
        Whether to evaluate the expansion to second degree; if set to False,
        a linear approximation will be returned instead (Default is True).
        
    Returns
    -------
    tl: float
        The approximated tempered likelihood.
    '''
    d = particle-mean
    A,B,C = Tcoefs
    if quadratic is False:
        approx = A+np.dot(B,d)
    else:
        d2 = np.power(d,2)
        approx = A+np.dot(B,d)+0.5*np.dot(C,d2)
    tl = coef*approx
    return(tl)

def test_approximation():
    '''
    Tests the analytical approximation in `approx_loglikelihood` by comparing 
    its result agains the exact value.
    Both are evaluated for the same representative set of inputs and printed on 
    the console.
    '''
    dim = glob.dim
    data = np.array([(random.random(),1),(random.random(),0)])
    mean = np.array([random.random() for d in range(dim)])
    coef = random.random()
    Tcoefs = Taylor_coefs(data,mean)
    max_dif = 0.1
    particle = np.array([x+random.uniform(-max_dif,max_dif) for x in mean])
    print("* Loglikelihood approximations:")
    print("- Exact: ",loglikelihood(data,particle,coef=coef))
    print("- Linear approximation: ", 
          approx_loglikelihood(particle,coef,
                               mean,Tcoefs,quadratic=False))
    print("- Quadratic approximation: ", 
          approx_loglikelihood(particle,coef,mean,Tcoefs,quadratic=True))
    
    print("* Log-gradient approximation:") 
    print("- Exact: ",loggradient(data,particle,coef=coef))
    print("- Approximation: ",approx_loggradient(particle,coef,mean,
                                               Tcoefs[1:]))
    print("[test_approximation]")
    
def loglikelihood_estimator(data, particle, coef, indices, mean, Tcoefs, 
                            control_variates=True, single_output=None):
    '''
    Provides an unbiased estimator to the (tempered) loglikelihood.
    
    Parameters
    ----------
    data: [([float],int)]
        The vector of data results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    particle: [float]
        The set of parameters at which the loglikelihood is to be evaluated.
    coef: float
        The tempering coefficient for the likelihood.
    indices: [float]
        The subsampling indices. 
    mean: [float]
        The mean particle, which defines the point around which the Taylor 
        expansion will be evaluated.
    Tcoefs: (float,float,float) 
        The 0-th to 2nd order Taylor coefficients.
    control_variates: bool, optional
        Whether to use control variates (Taylor approximations of each datum's
        likelihood, yielding a difference estimator with a smaller variance) 
        (Default is True).
    single_output: bool, optional
        Whether to output only the loglikelihood estimator or only its variance 
        (for testing purposes - namely automatic differentiation, because the 
        gradient calculator doesn't evaluate multiple output functions).
        
    Returns
    -------
    estimator: float
        The loglikelihood estimator.
    variance_estimator: float
        The loglikelihood variance estimator.
    '''
    measurements, samples = glob.measurements, glob.samples
    if not control_variates:
        subsample = [loglikelihood([data[i]],particle,coef=coef)
                 for i in indices]
        estimator = measurements/samples*sum(subsample)
        
        mean = sum(subsample)/len(subsample) # Don't use np.mean() because 
        #autograd doesn't support it and testing yields an error.
        squared_dev_mean = [(sample-mean)**2 for sample in subsample]
        variance_estimator = (measurements**2/samples**2)*sum(squared_dev_mean)
            
        if single_output=="estimator":
            return estimator
        if single_output=="variance":
            return variance_estimator
        return (estimator,variance_estimator)
            
    control_variates = approx_loglikelihood(particle,coef,mean,Tcoefs)
    deltas = [loglikelihood([data[i]],particle,coef=coef)
              -approx_loglikelihood(particle,coef,mean,
                                    Taylor_coefs([data[i]], mean))
              for i in indices]
    estimator = control_variates + measurements/samples*sum(deltas)
    if single_output=="estimator":
        return estimator
    
    mean_delta = sum(deltas)/len(deltas) 
    squared_dev_mean = [(delta-mean_delta)**2 for delta in deltas]
    variance_estimator  = (measurements**2/samples**2)*sum(squared_dev_mean)
    if single_output=="variance":
        return variance_estimator
    
    return (estimator,variance_estimator)

def likelihood_estimator(data, particle, coef, indices, mean, Tcoefs,
                         control_variates=True):
    '''
    Provides a slighlty biased estimator to the (tempered) likelihood.   
    The target is a perturbed distribution due to the fact that the variance 
    estimator is used in place of the actual population variance.
    
    Parameters
    ----------
    data: [([float],int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    particle: [float]
        The set of parameters at which the loglikelihood is to be evaluated.
    coef: float OR [float,float]
        The tempering coefficient (or a pair of, for the re-weighting ratio) 
        for the likelihood.
    indices: [float]
        The subsampling indices. 
    mean: [float]
        The mean particle, which defines the point around which the Taylor 
        expansion will be evaluated.
    Tcoefs: (float,float,float) 
        The 0-th to 2nd order Taylor coefficients.
    control_variates: bool, optional
        Whether to use control variates (Taylor approximations of each datum's
        likelihood, yielding a difference estimator with a smaller variance) 
        (Default is True).
        
    Returns
    -------
    l_estimator: float
        The (tempered) likelihood estimator.
    '''
    # Use coef 1 in computing the loglikelihood estimator to then eventually
    #use the pair.
    ll_est, ll_var_est = loglikelihood_estimator(data,particle,1,indices, 
                                mean,Tcoefs,control_variates=control_variates)
    if type(coef) is not list:
        l_estimator = np.exp(coef*ll_est-0.5*coef**2*ll_var_est)
    else:
        diff = coef[0]-coef[1]
        diff_squares = coef[0]**2-coef[1]**2
        l_estimator = np.exp(diff*ll_est-0.5*diff_squares*ll_var_est)        
    return l_estimator

def test_estimators(print_info=False):
    '''
    Tests the (tempered) likelihood and loglikelihood estimators from 
    `loglikelihood_estimator` and `likelihood_estimator` by comparing them
    against the exact computations of `loglikelihood` and `loglikelihood`
    respectively.
    Also evaluates a ratio of tempered likelihoods for some coefficients 
    `coef1` and `coef2` (1>`coef2`>`coef1`>0), to provide an idea of the order
    of magnitude of the error when re-weighting particles using the estimators.
    All are evaluated for the same representative set of inputs and printed on 
    the console.
    
    Parameters
    ----------
    print_info: bool, optional
        Whether to print the dummy information used for the calculations (data,
    etc.).
    '''
    prev_meas, prev_samples = glob.measurements, glob.samples
    glob.measurements, glob.samples = 10, 5 # For testing.
    dim, measurements, samples = glob.dim, glob.measurements, glob.samples
    t_max = 100

    #To test outcomes separately and not according to the actual likelihood:
    zero_outcomes = [(random.random()*t_max,0) for i in range(measurements//2)]
    one_outcomes = [(random.random()*t_max,1) for i in range(measurements//2)]
    data = np.concatenate([zero_outcomes,one_outcomes])
    '''
    ts = [t_max*random.random() for k in range(measurements)] 
    data=[(t,measure(t)) for t in ts]
    '''
    mean = np.array([random.random() for d in range(dim)])
    max_dev = 0.0001
    particle = np.array([x+random.uniform(-max_dev,max_dev) for x in mean])
    indices = [random.randrange(0,measurements) for m in range(samples)]
    coef, coef2 = random.uniform(0,1),random.uniform(0,1)
    coef, coef2 = max(coef,coef2), min(coef,coef2)
    
    Tcoefs = Taylor_coefs(data,mean)
    loglikelihood_estimator(data,particle,coef,indices,mean,Tcoefs) 
    if print_info:
        print("#Data: ",len(data),"\n#Subsample: ", samples, "\nMean: ",mean,
         "\nParticle:",particle,"\nIndices: ",indices,"\nCoefs: ", coef, coef2)
    
    print("* Loglikelihood estimator...")
    ll_est,ll_var = loglikelihood_estimator(data,particle,coef,indices,mean,
                                            Tcoefs,control_variates=True)
    print("- Using control variates:")
    print("  Estimator: ",ll_est)
    print("  Variance: ",ll_var)
    print("- No control variates:")
    ll_est2,ll_var2 = loglikelihood_estimator(data,particle,coef,indices,mean,
                                            Tcoefs,control_variates=False)
    print("  Estimator: ",ll_est2)
    print("  Variance: ",ll_var2)
    print("- Exact: ",loglikelihood(data,particle,coef=coef))
    
    print("* Likelihood estimator:")
    print("- Estimator (control variates): ", likelihood_estimator(data,
                    particle,coef,indices,mean,Tcoefs,control_variates=True))
    print("- Estimator (no control variates): ", likelihood_estimator(data,
                    particle,coef,indices,mean,Tcoefs,control_variates=False))
    print("- Exact: ", likelihood(data,particle,coef))
    
    print("* Ratio of tempered likelihoods:")
    print("- Estimator (control variates): ", 
          likelihood_estimator(data,particle,
                     [coef,coef2],indices,mean,Tcoefs,control_variates=True))
    print("- Estimator (no control variates): ", likelihood_estimator(data,
            particle,[coef,coef2],indices,mean,Tcoefs,control_variates=False))
    print("- Exact: ", likelihood(data,particle,coef-coef2))
    print("[test_estimators]")

    glob.measurements, glob.samples = prev_meas,prev_samples

def approx_loggradient(particle,coef,mean,Tcoefs):
    '''
    Provides an approximation to the gradient of the (tempered) loglikelihood 
    (using its Taylor series around a point).
    
    Parameters
    ----------
    particle: [float]
        The set of dynamical parameters to be used for the likelihood.
    coef: float
        The tempering coefficient to be used in computing the (annealed)
        likelihood.
    mean: [float]
        The mean particle, which defines the point around which the Taylor 
        expansion will be evaluated.
    Tcoefs: (float,float,float) 
        The 0-th to 2nd order Taylor coefficients.
        
    Returns
    -------
    approx_loggradient: [float]
        The approximated gradient of the loglikelihood.
    '''
    B,C = Tcoefs
    d = particle-mean
    approx_loggradient = coef*(B + np.dot(C,d))
    return approx_loggradient
    
def loglikelihood_estimator_gradient(data,particle,coef,indices,mean,Tcoefs,
                                     control_variates=True):
    '''
    Provides the gradients of the estimators of the (tempered) loglikelihood
    and its variance. Both are to be used in  computing the actual gradient 
    estimator.
    
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
    indices: [float]
        The subsampling indices. 
    mean: [float]
        The mean particle, which defines the point around which the Taylor 
        expansion will be evaluated.
    Tcoefs: (float,float,float) 
        The 0-th to 2nd order Taylor coefficients.
    control_variates: bool, optional
        Whether to use control variates (Taylor approximations of each datum's
        likelihood, yielding a difference estimator with a smaller variance) 
        (Default is True).
        
    Returns
    -------
    estimator_grad: float
        The gradient of the loglikelihood estimator.
    var_estimator_grad: float
        The gradient of the loglikelihood variance estimator.
    '''
    measurements, samples = glob.measurements, glob.samples
    if not control_variates:
        # Compute the gradient of the loglikelihood estimator.
        subsample_grad = [loggradient([data[i]],particle,coef) 
                          for i in indices]
        estimator_grad = measurements/samples*sum(subsample_grad)
        
        # Compute the gradient of the loglikelihood variance estimator.
        subsample = [loglikelihood([data[i]],particle,coef=coef)
                 for i in indices]
        mean = sum(subsample)/len(subsample) # Don't use np.mean() because 
        #autograd doesn't support it and testing yields an error.
        
        base = [sample-mean for sample in subsample]
        mean_grad = sum(subsample_grad)/samples 
        base_grad = [sample_grad-mean_grad for sample_grad in subsample_grad]
        var_estimator_grad = 2*measurements**2/samples**2\
            *sum([base_grad[i]*base[i] for i in range(samples)])
        return (estimator_grad,var_estimator_grad)
    
    # Compute the gradient of the loglikelihood estimator.
    control_variates = approx_loggradient(particle,coef,mean,Tcoefs[1:])
    delta_grads = [loggradient([data[i]],particle,coef)
              -approx_loggradient(particle,coef,mean,
                                    Taylor_coefs([data[i]], mean)[1:])
              for i in indices]
    
    estimator_grad = control_variates + measurements/samples*sum(delta_grads)
    
    # Compute the gradient of the loglikelihood variance estimator.
    deltas = [loglikelihood([data[i]],particle,coef=coef)
              -approx_loglikelihood(particle,coef,mean,
                                    Taylor_coefs([data[i]], mean))
              for i in indices]
    mean_delta = np.sum(deltas)/samples # Don't use np.mean() because autograd
    #doesn't support the implementation and testing yields an error.
    base = [delta-mean_delta for delta in deltas]
    mean_delta_grad = sum(delta_grads)/samples 
    base_grad = [delta_grad-mean_delta_grad for delta_grad in delta_grads]
    var_estimator_grad  = 2*measurements**2/samples**2\
        *sum([base_grad[i]*base[i] for i in range(samples)])
    return(estimator_grad,var_estimator_grad)

def loggradient_estimator(data,particle,coef,indices,mean,Tcoefs,
                          control_variates=True):
    '''
    Provides an estimator for the gradient of the (tempered) loglikelihood. 
    This matches the gradient of the loglikelihood associated with the 
    likelihood computed in `likelihood_estimator`.
    
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
    indices: [float]
        The subsampling indices. 
    mean: [float]
        The mean particle, which defines the point around which the Taylor 
        expansion will be evaluated.
    Tcoefs: (float,float,float) 
        The 0-th to 2nd order Taylor coefficients.
    control_variates: bool, optional
        Whether to use control variates (Taylor approximations of each datum's
        likelihood, yielding a difference estimator with a smaller variance) 
        (Default is True).
        
    Returns
    -------
    loggrad_est: float
        The estimator of the loglikelihood gradient.
    '''
    dlm,dsm = loglikelihood_estimator_gradient(data,particle,coef,indices,mean,
                                    Tcoefs,control_variates=control_variates)
    loggrad_est = dlm -0.5*dsm
    return loggrad_est

def test_gradient_estimator():
    '''
    Tests the (tempered) loggradient estimator from `loggradient_estimator` and 
    other quantities associated  with it, by comparing them against their 
    respective exact computations (namely that of `loggradient`).

    All are evaluated for the same representative set of inputs and printed on 
    the console.
    '''
    prev_meas,prev_samples = glob.measurements, glob.samples
    glob.measurements, glob.samples = 10, 5 # For testing; will be reset after.
    dim, measurements, samples = glob.dim, glob.measurements, glob.samples
    
    t_max = 100
    '''
    # To test outcomes separately and not according to the actual likelihood:
    zero_outcomes = [(random.random()*t_max,0) for i in range(measurements//2)]
    one_outcomes = [(random.random()*t_max,1) for i in range(measurements//2)]
    data = np.concatenate([zero_outcomes,one_outcomes])
    '''
    ts = [t_max*random.random() for k in range(measurements)] 
    glob.real_parameters = np.array([random.random() for d in range(dim)])
    data=[(t,measure(t)) for t in ts]
    
    mean = np.array([random.random() for d in range(dim)])
    
    max_dev = 0.001
    particle = np.array([x+random.uniform(-max_dev,max_dev) for x in mean])
    indices = [random.randrange(0,measurements) for m in range(samples)]
    coef = random.uniform(0,1)
    
    Tcoefs = Taylor_coefs(data,mean)
    
    est,var_est = loglikelihood_estimator_gradient(data,particle,coef,indices,
                                            mean,Tcoefs,control_variates=True)

    dlm = grad(loglikelihood_estimator,1)
    dlm_eval = dlm(data, particle, coef, indices, mean, Tcoefs,
                   single_output="estimator",control_variates=True)
    loggrad_est = loggradient_estimator(data,particle,coef,indices,mean,Tcoefs,
                                        control_variates=True)
    print("* Loglikelihood gradients...")
    print("- Using control variates:")
    print("  Gradient estimator: ",loggrad_est)
    print("  Calculated estimator gradient: ",est)
    print("  Exact estimator gradient: ",dlm_eval)
    print("  Gradient approximation: ",
          approx_loggradient(particle,coef,mean,Tcoefs[1:]))
    
    est2,var_est2 = loglikelihood_estimator_gradient(data,particle,coef,
                                    indices,mean,Tcoefs,control_variates=False)
    dlm2 = grad(loglikelihood_estimator,1)
    dlm_eval2 = dlm2(data, particle, coef, indices, mean, Tcoefs,
                   single_output="estimator",control_variates=False)
    loggrad_est2 = loggradient_estimator(data,particle,coef,indices,mean,
                                        Tcoefs,control_variates=False)
    print("- Without control variates:")
    print("  Gradient estimator: ",loggrad_est2)
    print("  Calculated estimator gradient: ",est2)
    print("  Exact estimator gradient: ",dlm_eval2)
    
    print("- Exact gradient: ",loggradient(data,particle,coef))
    
    dlm_eval = dlm(data, particle, coef, indices, mean, Tcoefs,
                   single_output="variance",control_variates=True)
    dlm_eval2 = dlm(data, particle, coef, indices, mean, Tcoefs,
                   single_output="variance",control_variates=False)
    print("* Variance gradients...")
    print("- Using control variates:")
    print("  Exact variance estimator gradient: ",dlm_eval)
    print("  Variance gradient estimator: ",var_est)
    print("- Without control variates:")
    print("  Exact variance estimator gradient: ",dlm_eval2)
    print("  Variance gradient estimator: ",var_est2)
    print("[test_gradient_estimator]")
    
    glob.measurements, glob.samples = prev_meas,prev_samples

def test_all():
    '''
    Calls all testing functions in the module.
    '''
    test_differentiation()
    print("")
    test_approximation()
    print("")
    test_estimators()
    print("")
    test_gradient_estimator()