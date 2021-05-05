# -*- coding: utf-8 -*-
"""
Performs inference on the frequencies of a multi-parameter binomial probability
distribution with success probability given by a sum of squared cosines: 
sum_i [ cos(theta_i*t/2)^2 ].

A sequential Monte Carlo approximation is used to represent the probability 
distributions, using Hamiltonian Monte Carlo and Metropolis-Hastings mutation 
steps.

The evolution times for the estimation are chosen offline, and picked according 
to some input parameters so as to tendentially increase them as the inference
process advances. 

Single or multiple runs can be performed; in the latter case information about 
the set of runs as a group is printed.
"""
reload = True
if reload:
    importlib.reload(sys.modules["global_vars"])
    importlib.reload(sys.modules["modules.likelihoods"])
    importlib.reload(sys.modules["modules.resampler"])
    importlib.reload(sys.modules["modules.distributions"])

import pprint, copy, itertools, random, importlib, matplotlib.pyplot as plt
from autograd import grad, numpy as np
import global_vars as glob
from modules.likelihoods import measure, likelihood, init_likelihoods
from modules.resampler import init_resampler, HMC_resampler, \
    print_resampler_stats, HMC_resampler_stats
from modules.distributions import SMCparameters, plot_distribution, \
    generate_prior, sum_distributions, init_distributions,  \
    mean_cluster_variance, cluster_stats

glob.dim = 2
glob.lbound = np.zeros(glob.dim) # The left boundaries for the parameters.
glob.rbound = np.ones(glob.dim) # The right boundaries for the parameters.

N_particles = None # Number of samples used to represent the probability
#distribution, using a sequential Monte Carlo approximation.

measurements = None # Length of the data record (coincides with the number of 
#Bayes updates to be performed in the case where chunksize=1).

real_parameters = None

lbound = np.zeros(glob.dim) # The left boundaries for the parameters.
rbound = np.ones(glob.dim) # The right boundaries for the parameters.

first_bayes_update = True
def bayes_update(data, new, distribution, threshold, signal_resampling=False,
                 allow_repeated=True):
    '''
    Updates a prior distribution according to the outcome of a measurement, 
    using Bayes' rule. 
    
    Parameters
    ----------
    data: [([float],int)]
        A vector of experimental results obtained so far and their respective 
        controls, each datum being of the form (time,outcome), where 'time' is          
        the control used for each experiment and 'outcome' is its result.
    new: slice
        The slice of the list `data` corresponding to freshly added data points
        (i.e. those which weren't contemplated in previous updates.
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
    allow_repeated: bool, optional
        Whether to allow repeated particles when resampling (Default is True).
        If False, extra Markov steps will be taken to ensure non-repetition when
        necessary, and alternative mutations will be used whem HMC is unlikely 
        to get an accepted proposal.
        
    Returns
    -------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The updated distribution (SMC approximation).
    resampled: bool
        Whether resampling has occurred.
    '''
    N_particles = glob.N_particles
    global first_bayes_update
    if first_bayes_update is True:
        rep = "; " if allow_repeated else "; not "
        rep += "allowing repeated particles"
        print("Bayes update: resampling threshold = ", threshold, rep)
        first_bayes_update = False

    acc_weight, acc_squared_weight = 0, 0
    
    # Perform a correction step by re-weighting the particles according to 
    #the last chunk of data added (i.e. to the ratio between the latest 
    #cumulative likelihood to the previous one, which cancels out all but those 
    #newest data).
    for key in distribution:
        particle = np.frombuffer(key,dtype='float64')
        new_weight = likelihood(data[new],particle)*distribution[key]
        distribution[key] = new_weight
        acc_weight += new_weight
    
    # Normalize the weights.
    for key in distribution:
        w = distribution[key]/acc_weight
        distribution[key] = w
        acc_squared_weight += w**2 # The inverse participation ratio will be
        #used to decide whether to resample.

    resampled = False
    if (1/acc_squared_weight <= threshold):
        resampled = True
        distribution,success = HMC_resampler(data, distribution, allow_repeated)
        if not success: 
            if signal_resampling:
                return distribution, None # Resampling interrupted midway.
            return distribution
    
    if signal_resampling:
        return distribution, resampled

    return distribution

first_offline_estimation = True
def offline_estimation(distribution, measurements, threshold=None, chunksize=10,
                        groupsize=20, increment=50, plot_all=False):
    '''
    Estimates the vector of parameters by sequentially performing experiments
    whose times (the controls) tendentially increase in a step by step manner.
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The prior distribution (SMC approximation).
    measurements: int
        The number of measurements/Bayes' updates to perform (these must match;
        each experiment's control is decided based on the latest posterior).
    threshold: float, optional
        The threshold effective sample size that should trigger a resampling 
        step when updating the distribution (Default is None, N_particles/2 
        will be used given the current value of the global variable 
        N_particles). 
    chunksize: int, optional
        The number of data to be added at each iteration; the cumulative data 
        makes up the target posterior to be sampled from at each step (Default 
        is 10).
    groupsize: int, optional
        The number of consecutive data whose maximum measurement time will 
        coincide (Default is 10).
    increment: float, optional
        The consecutive maximum measurement times that will be assigned to each
        group of #`groupsize` data; the actual times for individual data will be
        chosen at random with this value as upper cap (Default is 50).
    plot_all: bool, optional
        Whether to plot the particle positions at each step (Default is False).
        
    Returns
    -------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The final distribution (SMC approximation).
    data: [([float],int)]
        The vector of experimental results obtained and used for computing the
        target likelihoods the SMC algorithm sampled from. Each datum is of the 
        form (time,outcome), where 'time' is the control used for each 
        experiment and 'outcome' is its result.
    '''
    global first_offline_estimation
    if first_offline_estimation is True:
        print("Estimation: %d data; chunksize = %d; times = (i//%d+1)*%d"%
              (measurements,chunksize,groupsize,increment))
    
    if measurements==0:
        return
    
    if threshold is None:
        threshold = N_particles/2
        
    ans=""; resampled=False; counter=0; print("|0%",end="|")
    updates = measurements//chunksize
    data = []
    if updates < 10:
        progress_interval = 100/updates
    for i in range(updates):
        for j in range(chunksize):
            d = len(data) # Index of the datum to be added.
            # tmax=(g+1)*increment is a constant whithin each group g.
            g = (d//groupsize+1)
            tmax = (g+1)*increment
            #tmax = (g*2-1)*increment # To increase increment between groups.
            #prev_tmax = tmax - increment
            #time = random.randrange(prev_tmax,tmax)
            time = tmax*random.random() 
            data.append((time,measure(time)))

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
                plot_distribution(distribution,real_parameters, 
                                  note=info)
                
        # Signal newest data for SMC weight updates.
        new = slice(i*chunksize,(i+1)*chunksize) 
        # Update the distribution: get the posterior of the current iteration, 
        #which is the prior for the next.
        distribution, resampled = bayes_update(data[0:((i+1)*chunksize)], 
                          new, distribution, threshold, signal_resampling=True) 
        
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

        # Use the remaining data if applicable. This also ensures that all data 
        #will be used in a single update if the chunksize is greater than the 
        #total number of data. 
        left = measurements%chunksize
        if i==updates-1 and left!=0:
            for j in range(left):
                d = len(data) # Index of the datum to be added.
                tmax = (d//groupsize+1)*increment
                time = tmax*random.random() 
                data.append((time,measure(time)))

            new = slice(updates*chunksize,len(data))
            distribution = bayes_update(data, new, distribution, threshold) 
            
    print("") # For newline.

    if first_offline_estimation is True:
        print("Offline times: ", [t for t,o in data])
        first_offline_estimation = False
    return distribution, data

def run_single():
    '''
    Sets the settings for a run of the algorithm and performs inference.
    '''
    global real_parameters, N_particles, measurements
    dim = glob.dim
    glob.measurements = 100
        
    # To initialize some constants (for printing information on the console):
    init_distributions() 
    init_likelihoods()
    init_resampler() # Also for the global statistics.
    
    random_parameters = False
    if random_parameters:
        glob.real_parameters = np.array([random.random() for d in range(dim)])
    else:
        glob.real_parameters = np.array([0.25,0.77]) 
        #real_parameters = np.array([0.25,0.77,0.40])
        #real_parameters = np.array([0.25,0.77,0.40,0.52])
    
    # For ease of use since we don't want to change these variables anymore:
    real_parameters, measurements = glob.real_parameters, glob.measurements
    
    test_resampling, test_no_resampling = True, False

    glob.N_particles = 20**dim #'N_particles' particles will be used for each 
    #group. Should be a power with integer base and exponent `dim` so the 
    #particles can be neatly arranged into a cubic latice for the prior (unless 
    #not using a uniform distribution/lattice).

    prior = generate_prior(distribution_type="uniform")
    groups = 1 # The algorithm will be ran independently for `groups`
    #particle groups, on the same data. Their results will be joined 
    #together in the end.
    
    if test_resampling:
        final_dists = []
        for i in range(groups):
            print("~ Particle group %d (of %d) ~" % (i+1,groups))
            dist, data = offline_estimation(copy.deepcopy(prior),measurements,
                        chunksize=1,groupsize=30, increment=75,
                        threshold=100,plot_all=False)
            final_dists.append(dist)
            
        # To get the correct statistics:
        glob.N_particles = glob.N_particles*groups 
        final_dist = sum_distributions(final_dists) if groups>1 \
                                                    else final_dists[0]
        plot_distribution(final_dist,real_parameters)
        mean_cluster_variance(final_dist,real_parameters)

    glob.print_info()
    print_resampler_stats()
    
    if test_no_resampling: # For reference; use same data.
        global first_bayes_update, first_offline_estimation
        first_bayes_update,first_offline_estimation = True, False
        # Chunksize is irrelevant here so don't print it (hence the False ^)
        glob.N_particles = 50**dim
        prior = generate_prior(distribution_type="uniform")
        dist_no_resampling, data = offline_estimation(copy.deepcopy(prior),
                        measurements,
                        chunksize=1,groupsize=20, increment=100,
                        threshold=0,plot_all=False)
        glob.N_particles = 50**dim
        prior = generate_prior(distribution_type="uniform")
        dist_no_resampling = offline_estimation(copy.deepcopy(prior),data,
                                                threshold=0)
        plot_distribution(dist_no_resampling,real_parameters,
                          note="(no resampling)")
        print("No resampling test completed with %d particles." %
              glob.N_particles)
    
def run_several(nruns):
    '''
    Runs the estimation algorithm several times and reports back with the  
    relevant information.
    
    Parameters
    ----------
    nruns: int
        The number of runs to be performed.
    '''
    dim = glob.dim
    glob.measurements = 100
    glob.N_particles = 20**dim 
    prior = generate_prior(distribution_type="uniform")
    prox_thr, var_trh, unc_thr, covg_thr = 0.075, 0.025, 1.25, 95
    
    runs = []
    successful = []
    successful_dists = []
    successful_vars = []
    init_distributions() 
    run_number = 0
    while run_number!=nruns:
        first_err = True # For signaling first issue in 'space_occupation'.
        print("> Starting out run no. %d... [run_several]" % run_number)
        init_resampler(print_info=(True if run_number==0 else False))

        run = {}
        #glob.real_parameters = np.array([random.random() for d in range(dim)])
        glob.real_parameters = np.array([0.25,0.77]) 
        run["real_parameters"] = glob.real_parameters
        
        try:
            dist, data = offline_estimation(copy.deepcopy(prior),
                        glob.measurements,chunksize=1,groupsize=30,
                        increment=75,threshold=100,plot_all=False)
        except KeyboardInterrupt:
            print("> Keyboard interrupt, breaking from cycle... [run_several]")
            break
        except:
            print("> Error at run number ",run_number,": ", 
                  sys.exc_info()[0], " [run_several]")
            continue
        run["resampler_calls"],run["acceptance_ratio"] = HMC_resampler_stats()
        ts = [t for t,r in data]
        run["tmax"] = np.amax(ts)
        run["mean_var"], run["percent_dev"], run["distance"] = \
            cluster_stats(dist,glob.real_parameters)

        accuracy = run["distance"]<prox_thr
        precision = run["mean_var"]<var_trh
        correctness = run["distance"]<unc_thr*run["mean_var"]**0.5*dim
        mode_coverage = run["percent_dev"]<covg_thr
        
        success = accuracy and precision and correctness and mode_coverage 
        run["conditions (accuracy,precision,correctness,mode coverage)"] = \
            (accuracy,precision,correctness,mode_coverage)
        run["success"] = success
        if success: 
            successful.append(run_number)
            successful_vars.append(run["mean_var"])
            successful_dists.append(dist)
        succ = "[success]" if success else "[failed]"
        print("> Run %d completed. [run_several]" % run_number)
        if nruns<=20:
            plot_distribution(dist,run["real_parameters"],
                              note=("- run %d %s"% (run_number,succ)))
            pprint.pprint(run)
        runs.append(run)
        run_number+=1

    print("> Done. [run_several]")
    print("____________________")
    print("* Success rate: %d%%" % int(round(100*len(successful)/nruns)))
    if len(successful)>0:
        median_ind = successful_vars.index(np.percentile(successful_vars,50,
                                                       interpolation='nearest'))
        overall_ind = successful[median_ind]
        print("* Median variance among sucessful runs: ", 
              successful_vars[median_ind])
        print("* Mean variance among sucessful runs: ", 
              np.mean(successful_vars))
        print("* Run corresponding to the median variance (no. %d): " 
              % overall_ind)
        median_run = runs[overall_ind]
        pprint.pprint(median_run)
        plot_distribution(successful_dists[median_ind],
                        median_run["real_parameters"],
                        note=(": median variance successful run (no. %d) "
                        "[offline, t<100]" % overall_ind))
        
    print("_____________________________________________")
    glob.print_info()   

run_several(20)