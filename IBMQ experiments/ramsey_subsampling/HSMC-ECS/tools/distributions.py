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

def plot_particles(distribution, real_parameters, note=""):
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
        weights = [distribution[key][0]*200 for key in keys]
        axs.scatter(x1, x2, marker='o',s=weights)
        
    if dim%2!=0:
        # The last, unpaired dimension will be plotted alone with indices as x
        #values.
        global first_plot_particles
        if first_plot_particles is True:
            print("> Dimension is odd, cannot combine parameters pairwise; one"
                  " will be plotted alone.[plot_particles]")
        first_plot_particles = False
        
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
        particle_indices = [pair[0] for pair in particle_enum]
        particle_locations = [pair[1] for pair in particle_enum]
        
        weights = [distribution[key][0]*200 for key in keys]
        axs.scatter(particle_indices,particle_locations,s=weights)
 
        [axs.axhline(y=target, color='r',linewidth=0.75,linestyle="dashed") 
         for target in targets] 
        
def generate_prior(distribution_type = "uniform",
                   uniform_limits = (0,1)):    
    '''
    Creates a uniform, random (and assymptotically uniform), or 
    (assymptotically) gaussian discrete distribution on some region.
    Note that for SMC to actually be considering a non-uniform prior, the code 
    should be changed to accomodate that (i.e. it affects the likelihood, log-
    likelihood and log-likelihood gradient in ways that aren't cancelled out
    by virtue of the effect being identical for all particles).
    
    Parameters
    ----------
    distribution_type: str, optional
        The class of distribution to be used (should be uniform, random or 
        gaussian/normal)
    uniform_limits: (float,float), optional
        The left and right boundaries if the prior distribution is to be flat
        (Default is (0,1)). These are assumed to be the same for every 
        dimension/parameter.
        
    Returns
    -------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        A dictionary representing the requested distribution (SMC 
        approximation).
    '''

    N_particles, measurements, samples, dim = \
        glob.N_particles, glob.measurements, glob.samples, glob.dim
    lbound, rbound = np.array([uniform_limits[0] for d in range(dim)]),\
        np.array([uniform_limits[1] for d in range(dim)])
    #lbound, rbound =  glob.lbound, glob.rbound # To cover all space.
    if distribution_type=="uniform":
        # Number of particles along each dimension:
        each = int(glob.N_particles**(1/dim)) 
        # Parameter spans:
        width = [rbound[i]-lbound[i] for i in range(dim)] 
        
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
        
    print("[Generated %s prior distribution" % (distribution_type),
          end="")
    extra_info_flat = " (between %.1f and %.1f)]." % uniform_limits
    print(extra_info_flat if distribution_type=="uniform" else "].")
        
    # Convert list to dictionary with keys as particles and uniform weights
    #(the distribution is characterized by particle density).
    prior = {}
    for particle in particles:
        particle = np.asarray(particle)
        # Use bit strings as keys because numpy arrays are not hashable.
        key = particle.tobytes()
        # Choose subsampling indices.
        u = [random.randrange(0,measurements) for m in range(samples)]
        prior[key] = [1/N_particles,u]
        
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
    #for each key. Absence means of course a weight of 0, and the subsampling
    #indexes can be any (including empty lists) because by this point the 
    #estimation will be complete.
    final_dist = {key: [np.sum([d.get(key, [0,[]])[0] for d in distributions])/
                  len(distributions),[]] for key in all_keys}
    return final_dist
    
    