# -*- coding: utf-8 -*-
"""
Script to generate a 3-d plot of the kernel density estimate for a set of 2-d  
points imported from a file.
"""
import pickle, matplotlib.pyplot as plt
from autograd import grad, numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab
dim=2

def multivariate_gaussian(x, mu, sigma, normalize=False):
    x_mu = np.array(x-mu)
    sigma_inv = np.linalg.inv(sigma)
    power = -0.5*np.linalg.multi_dot([x_mu,sigma_inv,x_mu])

    k = len(x)
    sigma_det = np.linalg.det(sigma)
    norm = ((2*np.pi)**k*sigma_det)**0.5
    if normalize:
        return(np.exp(power))
    else:
        return(np.exp(power)/norm)
    
def kernel_density_estimate(x, points):
    n = len(points)
    h = n**-0.2 # Bandwidth.
    kde = 0
    for point in points:
        kde += multivariate_gaussian(x,point,h*np.identity(dim),
                                    normalize=True)/(n*h)
    return(kde)

def simple_scatter(points):
    fig, axs = plt.subplots(1,figsize=(8,8))
    axs.scatter([point[0] for point in points],[point[1] for point in points],
                marker='o',s=8)
    plt.xlim((-6,6))
    plt.ylim((-3,30))

def main():
    with open('smileyface2.data', 'rb') as filehandle:
        points = pickle.load(filehandle)
    simple_scatter(points[len(points)-1]) # The list items are cumulative data-
    #chunks, so the last item contains all points.
        
    xx = np.linspace(-6, 6, 50)
    yy = np.linspace(-3, 30, 50)
    zz = np.zeros((xx.size,yy.size))
    i1 = 0
    for y in yy:
        i2 = 0
        for x in xx:
            zz[i2,i1] = kernel_density_estimate(np.array([x,y]),
                                                points[len(points)-1])
            i2 += 1
        i1 += 1
        
    X, Y = np.meshgrid(yy, xx)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(elev=30, azim=120)
    ax.plot_surface(X, Y, zz)
    plt.show()

    
main()