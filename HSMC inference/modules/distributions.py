# -*- coding: utf-8 -*-
"""
Module with SMC distribution-related functions.
"""
import itertools, random
import numpy as np, matplotlib.pyplot as plt
import global_vars as glob

first_plot_particles = True

def init_distributions():
    '''
    Initializes constants pertaining to the module.
    '''
    global first_plot_particles
    first_plot_particles = True
    

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
    dim,N_particles = glob.dim, glob.N_particles
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
    dim, lbound, rbound = glob.dim, glob.lbound, glob.rbound
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
    N_particles, measurements, dim = \
        glob.N_particles, glob.measurements, glob.dim
    lbound, rbound =  glob.lbound, glob.rbound
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
        
    print("[Generated %s prior distribution.]" % (distribution_type))
    return(prior)
        
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
    
def split_dict(distribution):
    '''
    Converts a dictionary representation of a distribution to a tuple 
    representation, with a list giving the particle locations and another the
    weights (by the same order).
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The distribution to be converted (SMC approximation).
        
    Returns
    -------
    particles: [[float]]
        A list of particle locations.
    weights: [float]
        A list of particle weights, by the same order as the locations.
    '''
    particles,weights = [],[]
    for key in distribution:
        particle = np.frombuffer(key,dtype='float64')
        weight = distribution[key]
        particles.append(particle)
        weights.append(weight)
    return particles,weights

def mean_weighted_variance(points, weights):
    '''
    Computes the mean over dimensions of the variance of a weighted set of 
    points.
    
    Parameters
    ----------
    points: [[float]]
        The list of points (ordered coordinates).
    weights: [float]
        The list of particle weights, by the same order as in 'particles'.
        
    Returns
    -------
    mean_var: float
        The average variance over all dimensions.
    '''
    dim = glob.dim
    acc = 0
    for d in range(dim):
        coord = [point[d] for point in points]
        average = np.average(coord, weights=weights)
        variance = np.average((coord-average)**2, weights=weights)
        acc += variance
    mean_var = acc/dim
    return (mean_var)

def mean_cluster_variance(distribution,real_parameters):
    '''
    Computes and prints the mean over clusters and dimensions of the variance of 
    a distribution, and the mean absolute difference between the number of 
    particles around each mode and its average.
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The distribution to be converted (SMC approximation).
    real_parameters: [float]
        The real parameters that define the target modes (it suffices to take 
        all permutations). They will be used as centroids when clustering.
    '''
    particles,weights = split_dict(distribution)
    modes = list(itertools.permutations(real_parameters))
    ngroups = len(modes)
    grouped_particles = [[] for i in range(ngroups)]
    grouped_weights = [[] for i in range(ngroups)]
    # Group the particles by mode.
    for i in range(len(particles)):
        distances = [np.linalg.norm(particles[i]-mode) for mode in modes]
        mode = np.argmin(distances)
        grouped_particles[mode].append(particles[i])
        grouped_weights[mode].append(weights[i])

    vars = [mean_weighted_variance(grouped_particles[i],grouped_weights[i]) \
            if len(grouped_particles[i])!=0 else 0
            for i in range(ngroups)]
    mean_var = np.mean(vars)
    print("Mean variance over clusters and dimensions: ", mean_var)
    particles_per_mode = [len(grouped_particles[i]) for i in range(ngroups)]
    mean = np.mean(particles_per_mode)
    mean_dev = np.mean([abs(np-mean) for np in particles_per_mode])
    print("Mean percentual deviation of the particle number per mode: %d%%" % 
          round(100*mean_dev/mean))
          
def space_occupation(points,side=50,thr=0.1):
    '''
    Estimates the fraction of space occupied by some group of points, by 
    splitting the volume into dim-dimensional cubes and counting how many 
    contain at least one point.
    
    Parameters
    ----------
    points: [[float]]
        The list of coordinate vectors of the points.
    side: int, optional
        The dim-dimensional root of the number of cells into which to split 
        the space (Default is 50).
    thr: float, optional
        The threshold fraction of particles in different cells that should
        trigger a doubling of the 'side' parameter.
        
    Returns
    -------
    r: float
        The estimated space occupation ratio.
    side: int
        The maximum side used.
    '''
    dim, N_particles = glob.dim, glob.N_particles
    ncells = side**dim
    occ = set()
    cell_width = 1/side
    for point in points:
        cell = tuple(int(point[i]//cell_width) for i in range(dim))
        occ.add(cell)
    r = len(occ)/ncells
    if len(occ)<=thr*N_particles: 
        # If not at least thr% of particles in different cells, fine grain cells 
        #further to get a better estimate (occupation ratio will tend to be 
        #smaller).
        space_occupation(points,side=10*side)
    return r,side