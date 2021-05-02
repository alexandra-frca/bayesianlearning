'''
Provides a quantitative estimate of the uncertainty of a multimodal and
multi-dimensional distribution by clustering the particles around each mode,
computing the variance along each dimension and mode, and then taking the 
average over both.
'''

import numpy as np, itertools

dim = 2
N = 1000
real_parameters = np.array([0.2,0.8])
modes = list(itertools.permutations(real_parameters))

cov = 0.01*np.identity(dim)
particles = []
weights = []
for mode in modes:
    for i in range(N//2):
        particles.append(np.random.multivariate_normal(mode,cov))
        weights.append(1)

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
    acc = 0
    for d in range(dim):
        coord = [point[d] for point in points]
        average = np.average(coord, weights=weights)
        variance = np.average((coord-average)**2, weights=weights)
        acc += variance
    mean_var = acc/dim
    return (mean_var)

def mean_cluster_variance(particles,weights,real_parameters):
    '''
    Computes and prints the mean over clusters and dimensions of the variance of 
    a weighted set of points, and the mean absolute difference between the 
    number of particles around each mode and its average.
    
    Parameters
    ----------
    particles: [[float]]
        The list of points (ordered coordinates).
    weights: [float]
        The list of particle weights, by the same order as in 'particles'.
    real_parameters: [float]
        The real parameters that define the target modes (it suffices to take 
        all permutations). They will be used as centroids when clustering.
    '''
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
            for i in range(ngroups)]
    mean_var = np.mean(vars)
    print("Mean variance over clusters and dimensions: ", mean_var)
    particles_per_mode = [len(grouped_particles[i]) for i in range(ngroups)]
    mean = np.mean(particles_per_mode)
    mean_dev = np.mean([abs(np-mean) for np in particles_per_mode])
    print("Mean absolute deviation of the particle number per mode: ",mean_dev)

mean_cluster_variance(particles,weights,real_parameters)