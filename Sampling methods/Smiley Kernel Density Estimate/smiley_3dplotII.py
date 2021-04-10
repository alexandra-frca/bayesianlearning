# -*- coding: utf-8 -*-
"""
Script to generate a 3-d plot of the kernel density estimate for a set of 2-d  
points imported from a file.
"""
import pickle, matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
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

def plot_3d(points):
    xx = np.linspace(-6, 6, 50)
    yy = np.linspace(-3, 30, 100)
    zz = np.zeros((xx.size,yy.size))

    for i1,y in enumerate(yy):
        for i2,x in enumerate(xx):
            zz[i2,i1] = kernel_density_estimate(np.array([x,y]),
                                                points[len(points)-1])
        
    X, Y = np.meshgrid(xx, yy)

    fig = plt.figure(figsize=(16,16))
    ax = Axes3D(fig)
    
    zmax = 0.5
    levels = np.linspace(0, zmax, 500)
    ax.plot_wireframe(X, Y, zz.T, rstride=1, cstride=1, color='gray')
    ax.contourf(X, Y, zz.T,
                zdir='z', levels=levels, alpha=0.7)
    #ax.view_init(elev=30, azim=120)
    

    
    xs = [point[0] for point in points[len(points)-1]]
    ys = [point[1] for point in points[len(points)-1]]
    zs = [kernel_density_estimate(np.array([xs[i],ys[i]]),
                                                points[len(points)-1])
          for i in range(len(xs))]
    #w = [10**2 if xs[i]>0 else 6**2 for i in range(len(xs))]
    w = [6**2 for i in range(len(xs))]
    ax.scatter(xs, ys,zs, c='k',s=w)
    plt.show()
    
def main():
    with open('smileyface2.data', 'rb') as filehandle:
        points = pickle.load(filehandle)
        
    points=[points[len(points)-1]] # To use less data.
    points=[points[0]] # To use less data.

    plot_3d(points)
main()