# -*- coding: utf-8 -*-
"""
Module with statistics-related functions.
"""
import numpy as np, matplotlib.pyplot as plt
import global_vars as glob

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
    e = np.exp(power)
    if not normalize:
        return e
    norm = (2*np.pi*sigma**2)**0.5  
    e = e/norm
    return e

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
    stdev: bool OR str, optional 
        To be set to False if the standard deviation is not to be returned, and 
        to "corrected" if the corrected sample standard deviation is intended
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
            w = distribution[key][0]
        means += particle*w
        meansquares += particle**2*w
    if not stdev:
        return means
    stdevs = abs(means**2-meansquares)**0.5
    if stdev=="corrected": # Which also evaluates to True (non empty string).
        # Get the corrected sample standard deviation, which yields un unbiased 
        #variance.
        stdevs = stdevs*(N_particles/(N_particles-1))**0.5
    return means,stdevs

def print_info():
    '''
    Prints some relevant information relative to the run of the algorithm on
    the console.
    '''
    N_particles, measurements, samples, dim = \
        glob.N_particles, glob.measurements, glob.samples, glob.dim
        
    print("> n=%.2f^%d; N=%d; m=%d; %dd sum of squared cosines " 
          "(possibly with subsampling, HSMC-ECS)" % 
          (N_particles**(1/dim),dim,measurements,samples,dim))
    
def dict_to_list(distribution):
    '''
    Converts a dictionary representation of a distribution to a list-of-tuples
    representation.
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The distribution to be converted (SMC approximation).
        
    Returns
    -------
    particles: [([float],float)]
        A list of (particle,weight) tuples, where `particle` is a parameter 
        vector.
    '''
    particles = []
    for key in distribution:
        particle = np.frombuffer(key,dtype='float64')
        weight = distribution[key][0]
        particles.append((particle,weight))
    return particles
    
def kernel_density_estimate(x, points, stdev):
    '''
    Computes the value of a kernel density estimate at some point.
    
    Parameters
    ----------
    x: float 
        The point at which the estimate should be evaluated. 
    points: [[float]]
        A list of coordinate vectors denoting the point locations.
    stdev: float 
        The standard deviation of the points, to be used to choose the 
        bandwidth for the KDE.
        
    Returns
    -------
    kde: float
        The value of the kernel density estimate.
    '''
    n = len(points)
    h = 1.06*stdev*n**-0.2
    kde = 0
    for point in points:
        kde += gaussian(x,point,h,normalize=True)/(n*h)
    return(kde)

def weighted_kernel_density_estimate(x, points, stdev):
    '''
    Computes the value of a (weighted) kernel density estimate at some point.
    
    Parameters
    ----------
    x: float 
        The point at which the estimate should be evaluated. 
    points: [([float],float)]
        A list of (point,weight) tuples, where `particle` is a 1-d coordinate 
        vector.
    stdev: float 
        The standard deviation of the points, to be used to choose the 
        bandwidth for the KDE.
        
    Returns
    -------
    kde: float
        The value of the kernel density estimate.
    '''
    nonzero = [p for p in points if p[1]>0]
    n = len(nonzero)
    h = 1.06*n**-0.2*stdev
    kde = 0
    for point in points:
        p,w = point
        kde += gaussian(x,p,h,normalize=True)/n
        # Division by h is already accounted for by the normalization.
    return(kde)

def plot_scatter(points):
    '''
    Plots a curve given a list of x-coordinates and the corresponding list of
    y-coordinates.
    
    Parameters
    ----------
    points: [(float,float)]
        The list of coordinate pairs of the points to be plotted.
    '''
    fig, axs = plt.subplots(1,figsize=(8,8))
    axs.scatter([point[0] for point in points],[point[1] for point in points],
                marker='o',s=5)
    plt.xlim((0,1))
    plt.ylim((0,1))

def plot_curve(xs,ys):
    '''
    Plots a curve given a list of x-coordinates and the corresponding list of
    y-coordinates.
    
    Parameters
    ----------
    xs: [float]
        The first coordinates of the points to make up the curve.
    ys: [float]
        The second coordinates of the points to make up the curve, by the same
        order as in `xs`.
    '''
    fig, axs = plt.subplots(1,figsize=(8,8))
    axs.plot(xs,ys,color="black",linewidth=1,linestyle="dashed",
             label="Data points")
    axs.legend(loc='upper right',fontsize=14)
    plt.xlim((0,1))
    plt.ylim((0,1))

def plot_kde(distribution):
    '''
    Computes and plots the kernel density estimate of a distribution.
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The distribution of interest, to be represented by a KDE.
    '''
    stdev = SMCparameters(distribution,stdev="corrected")[1]
    ks =  dict_to_list(distribution)
    xs = np.arange(0,2,0.05)
    ys = [weighted_kernel_density_estimate(x,ks,stdev) for x in xs]
    plot_curve(xs,ys)
    
def plot_kdes(distribution,reference,labels):
    '''
    Computes and plots the kernel density estimates of 2 distributions for 
    their comparison.
    
    Parameters
    ----------
    distribution: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        The distribution of interest, to be represented by a KDE.
    reference: dict
        , with (key,value):=(particle,importance weight) 
        , and particle the parameter vector (as a bit string)
        A distribution whose KDE is to be plotted for comparison (with dashed 
        lines).
    labels: [str] 
        The ordered labels to be assigned to `distribution` and `reference`, by
        this order. 
    '''
    stdev = SMCparameters(distribution,stdev="corrected")[1]
    rstdev = SMCparameters(reference,stdev="corrected")[1]
    print("- Standard deviation (subsampling):",stdev)
    print("- Reference (full data): ", rstdev)
    avg_stdev = (stdev+rstdev)/2
    print("[plot_kdes]")
    ks = dict_to_list(distribution)
    rks = dict_to_list(reference)
    xs = np.arange(0.7,0.9,0.001)
    ys =  [weighted_kernel_density_estimate(x,ks,avg_stdev) for x in xs]
    rys = [weighted_kernel_density_estimate(x,rks,avg_stdev) for x in xs]
    
    fig, axs = plt.subplots(1,figsize=(8,8))
    axs.plot(xs,ys,color="red",linewidth=1,label=labels[0])
    axs.plot(xs,rys,color="black",linewidth=1,linestyle="dashed",
             label=labels[1])
    axs.set_title("Kernel density estimates")
    axs.legend(loc='upper left',fontsize=14)