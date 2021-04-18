# -*- coding: utf-8 -*-
"""
Performs inference on the frequencies of a multi-parameter binomial probability
distribution with success probability given by a sum of squared cosines: 
sum_i [ cos(theta_i*t/2)^2 ], using tempered likelihood SMC and optionally 
subsampling a fraction of the observations (for both the weight updates and 
resampling steps).

A sequential Monte Carlo approximation is used to represent the probability 
distributions, using Hamiltonian Monte Carlo and Metropolis-Hastings mutation 
steps.

If subsampling, "energy conserving subsampling HMC" is used; this is achieved 
by wrapping the HMC transitions in a Gibbs update, in which a Metropolis-
-Hastings step is used to pick the subsampling indices (conditioned on the 
latest particle location) before the usual HMC (or gaussian-proposal MH) step 
(conditioned on the indices).

This amounts to a pseudo-marginal approach where the posterior is augmented to
include some auxiliary variables, in this case are the subsampling indices. The
extended target density (encompassing the indices) is left invariant by  the
"composite" Markov transition.

The HMC moves preserve the energy because  the MH acceptance/rejection step 
targets precisely the energy according to which the Hamiltonian dynamics are 
simulated, guaranteeing correctness. As such, marginalizing over the indices 
should yield parameter samples that abide by the target distribution.

For the subsampling indices, correlation is introduced between consecutive
proposals by updating only a segment of the previous index set (correlated/
block pseudo-marginal sampler). The purpose is to make the acceptance rate
more stable, preventing the "locking" of some set of indices that could happen 
otherwise (should the estimator significantly overestimate the likelihood 
associated with them, yielding a too low acceptance probability for all 
others).

The target distributions consist of a sequence of annealed likelihoods, i.e. 
the likelihood resulting from the whole data set, raised to the power of some 
annealing coefficients.

Estimators are used for the likelihood, optionally using control variates 
(though they are Taylor approximation based and only work under unimodality,
and so aren't suitable for the chosen target if the parameter vector dimension 
is >=2 unless its components are by the model's construction all the same).

The particle positions are plotted, as well as kernel density estimates for the 
final full data and subsampling distributions if both strategies are performed
and the parameter vector is one-dimensional.

Based on "Hamiltonian Monte Carlo with Energy Conserving Subsampling"
[https://arxiv.org/pdf/1402.4102.pdf]
, "Subsampling Sequential Monte Carlo for Static Bayesian Models"
[https://arxiv.org/pdf/1805.03317.pdf]
and "The Block Pseudo-Marginal Sampler"
[https://arxiv.org/pdf/1603.02485.pdf]
"""

import importlib, sys, copy
import random, numpy as np
import global_vars as glob
from function_evaluations.likelihoods import measure, likelihood, \
    init_likelihoods
from function_evaluations.estimators import Taylor_coefs,likelihood_estimator
from tools.resampler import init_resampler, HMC_resampler, \
    print_resampler_stats
from tools.statistics import SMCparameters, print_info, plot_kdes
from tools.distributions import plot_particles, generate_prior, \
    sum_distributions, init_distributions
#np.seterr(all='warn')

reload = False
if reload:
    importlib.reload(sys.modules["global_vars"])
    importlib.reload(sys.modules["function_evaluations.likelihoods"])
    importlib.reload(sys.modules["function_evaluations.estimators"])
    importlib.reload(sys.modules["tools.resampler"])
    importlib.reload(sys.modules["tools.statistics"])
    importlib.reload(sys.modules["tools.distributions"])

glob.dim = 1
glob.lbound = np.zeros(glob.dim) # The left boundaries for the parameters.
glob.rbound = np.ones(glob.dim) # The right boundaries for the parameters.
        
first_bayes_update = True
def bayes_update(data, distribution, coef, previous_coef, threshold,
                 subsample, control_variates, signal_resampling=False):
    '''
    Updates a prior distribution according to the outcome of a measurement, 
    using Bayes' rule. 
    
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
    threshold: float
        The threshold effective sample size that should trigger a resampling 
        step. 
    subsample: bool
        Whether to subsample.
    control_variates: bool
        Whether to use control variates (Taylor approximations of each datum's
        likelihood, yielding a difference estimator with a smaller variance).
    signal_resampling: bool
        Whether to return a second variable denoting the ocurrence of 
        resampling.    
        
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

    acc_weight, acc_squared_weight = 0, 0
    mean = SMCparameters(distribution,stdev=False)
    Tcoefs = Taylor_coefs(data,mean)
    # Perform a correction step by re-weighting the particles according to 
    #the last chunk of data added (i.e. to the ratio between the latest 
    #cumulative likelihood to the previous one, which cancels out all but those 
    #newest data).
    for key in distribution: 
        particle = np.frombuffer(key,dtype='float64')
        weight,u = distribution[key]
        if subsample:
            L = likelihood_estimator(data,particle,[coef,previous_coef],u,mean,
                                     Tcoefs,control_variates=control_variates)
        else: 
            L = likelihood(data,particle,coef=coef-previous_coef)
            
        new_weight = L*weight
        distribution[key][0] = new_weight
        acc_weight += new_weight

    if acc_weight==0:
        print("> All zero weights after update. [bayes_update]")
        
    # Normalize the weights.
    for key in distribution:
        w = distribution[key][0]/acc_weight
        distribution[key][0] = w
        acc_squared_weight += w**2 # The inverse participation ratio will be
        #used to decide whether to resample.
        
    resampled = False
    if (1/acc_squared_weight <= threshold):
        resampled = True
        distribution, success = HMC_resampler(data, distribution, coef, 
                              subsample, control_variates=control_variates)
        if not success: 
            if signal_resampling:
                return distribution, None # Resampling interrupted midway.
            return distribution
    
    if signal_resampling:
        return distribution, resampled

    return distribution

first_offline_estimation = True
def offline_estimation(distribution, measurements, tmax , 
                       tempering_coefficients, threshold=None, subsample=False,
                       control_variates=True, plot_all=False):
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
    measurements: int
        The number of measurements/experiment to be performed.
    tmax: float
        The maximum measurement time to be used; each actual time will be 
        chosen at random between 0 to this value.
    tempering_coefficients: float
        The sequence of tempering coefficients to be used in computing the 
        annealed likelihoods.
    threshold: float, optional
        The threshold effective sample size that should trigger a resampling 
        step when updating the distribution (Default is None, N_particles/2 
        will be used given the current value of the global variable 
        N_particles). 
    subsample: bool, optional
        Whether to subsample (Defalut is False).
    control_variates: bool, optional
        Whether to use control variates (Taylor approximations of each datum's
        likelihood, yielding a difference estimator with a smaller variance)
        (Defalut is False). This is not used unless subsample=True.
    plot_all: bool, optional
        Whether to plot the particle positions at each step (Default is False).
        
    Returns
    -------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The final distribution (SMC approximation).
    '''
    N_particles, real_parameters, measurements, samples = \
        glob.N_particles, glob.real_parameters, glob.measurements, glob.samples
        
    if measurements==0 or len(tempering_coefficients)==0:
        return
        
    global first_offline_estimation
    if first_offline_estimation is True:
        print("Offline estimation: random times <= %d" % tmax)
        print("Tempering coefficients: ",tempering_coefficients)
        cv = "(with control variates)." if control_variates else \
            "(without control variates)."
        info = ("subsampling %d/%d observations " % (samples,measurements))\
            + cv if subsample else "full data."
        print("Estimation - ",info)
        first_offline_estimation = False
    
    if threshold is None:
        threshold = N_particles/2
        
    ts = [tmax*random.random() for k in range(measurements)] 
    data=[(t,measure(t)) for t in ts]
        
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
                info += (" (subsampled %d/%d observations)" % 
                         (samples,measurements)) if subsample else ""
                info += " [resampled]" if resampled else ""
                plot_particles(distribution,real_parameters,note=info)
            
        # Update the distribution: get the posterior of the current iteration, 
        #which is the prior for the next.
        distribution, resampled = bayes_update(data, distribution, coef, 
                tempering_coefficients[i-1], threshold, subsample,
                control_variates, signal_resampling=True) 
        if resampled is None:
            print("> Process interrupted at iteration %d due to non-invertible"
                  " covariance matrix. [offline_estimation]" % i,end="")
            if plot_all:
                plot_particles(distribution,real_parameters, 
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
    global first_bayes_update, first_offline_estimation
    dim = glob.dim
    random_parameters = False
    if random_parameters:
        glob.real_parameters = np.array([random.random() 
                                         for d in range(dim)])
    else:
        glob.real_parameters = np.array([0.8]) 
        #glob.real_parameters = np.array([0.25,0.77]) 
        #glob.real_parameters = np.array([0.25,0.77,0.40,0.52])
    
    glob.measurements = 100
    glob.samples = 20
    # For ease of use since we don't want to change these variables anymore:
    real_parameters, measurements, samples = \
        glob.real_parameters, glob.measurements, glob.samples
        
    # To initialize some constants (for printing information on the console):
    init_distributions() 
    init_likelihoods()
    init_resampler() # Also for the global statistics.
    
    steps = 1
    coefs = [i/steps for i in range(steps+1)] # These coefficients could be
    #chosen adaptively to keep the effective sample size near some target value
    #, but evenly spaced coefficients seem to work well for this problem and 
    #don't require extra Bayes updates. 
    #coefs=coefs[0:2]

    tmax = 100 # If tmax is too large, subsampling works poorly (because the
    #likelihood is too sharp, the derivatives too large, and the approximations
    #and estimators very off-target).
    
    test_resampling, test_no_resampling = True, False
    test_subsampling, test_no_resampling_subsampling = True, False
    
    glob.N_particles = 100**dim #`N_particles` particles will be used for each 
    #group. Should be a power with integer base and exponent `dim` so the 
    #particles can be neatly arranged into a cubic latice for the prior (unless 
    #not using a uniform distribution/lattice).
    prior = generate_prior(distribution_type="uniform")
    ngroups = 1
    
    if test_no_resampling: # Just for reference.
        dist_no_resampling = offline_estimation(copy.deepcopy(prior),
                         measurements, tmax, coefs,threshold=0,plot_all=False)
        plot_particles(dist_no_resampling,real_parameters,
                          note="(no resampling)")
        print("> No resampling test completed.")
        
    if test_no_resampling_subsampling: 
        dist_no_resampling_subs = offline_estimation(copy.deepcopy(prior),
                                              coefs,threshold=0,
                                              subsample=True,plot_all=False)
        plot_particles(dist_no_resampling_subs,real_parameters,
                    note=(" (no resampling; subsampled %d/%d observations)" 
                    % (samples,measurements)))
        print("> No resampling test with subsampling completed.")
    
    if test_resampling:
        first_bayes_update, first_offline_estimation = True, True
        
        print("> Testing full data HSMC...")
        
        groups = ngroups # The algorithm will be ran independently for `groups`
        #particle groups, on the same data. Their results will be joined 
        #together in the end.
        final_dists = []
        for i in range(groups):
            print("~ Particle group %d (of %d) ~" % (i+1,groups))
            final_dists.append(offline_estimation(copy.deepcopy(prior),
              measurements,tmax, coefs,threshold=float('inf'),plot_all=False))
        # To get the correct statistics, get the total particles.
        if groups > 1:
            glob.N_particles = glob.N_particles*groups 
            final_dist = sum_distributions(final_dists)
        else:
            final_dist = final_dists[0]
        plot_particles(final_dist,real_parameters)
        
    if test_subsampling:
        print("> Testing subsampling...")
        first_bayes_update, first_offline_estimation = True, True
        groups = ngroups
        #glob.N_particles = 30**dim 
        #prior = generate_prior(distribution_type="uniform")
        warmup = False
        dist = copy.deepcopy(prior)
        if warmup:
            # If using, adapt sequential weight updates to account for this
            #when transitioning (estimator of all data)**(first real coef)
            #dividing by (estimator of warm-up data)**(last warm up coef).
            print("> Warming up...")
            warmup_tmax = 1
            #warmup_coefs = [0,0.2]
            dist = offline_estimation(dist,measurements, warmup_tmax, coefs,
                        threshold=float('inf'),subsample=True,plot_all=False)
            first_bayes_update, first_offline_estimation = True, True
            plot_particles(dist,real_parameters, 
                          note=(" (subsampled %d/%d observations; warm-up)" % 
                         (samples,measurements)))
            
        final_dists_subs = []
        for i in range(groups):
            print("~ Particle group %d (of %d) ~" % (i+1,groups))
            final_dists_subs.append(offline_estimation(dist,
                        measurements, tmax, coefs,threshold=float('inf'),
                        subsample=True,plot_all=False))
        if groups > 1:
            glob.N_particles = glob.N_particles*groups 
            final_dist_subs = sum_distributions(final_dists_subs)
        else:
            final_dist_subs = final_dists_subs[0]
        plot_particles(final_dist_subs,real_parameters, 
                          note=(" (subsampled %d/%d observations)" % 
                         (samples,measurements)))
        
    if dim==1 and test_resampling and test_subsampling:
        plot_kdes(final_dist,final_dist_subs,
                 labels=["Full data HSMC","Subsampling HSMC"])

    print_info()
    print_resampler_stats()

main()