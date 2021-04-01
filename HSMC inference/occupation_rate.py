# -*- coding: utf-8 -*-
"""
Approximating the parameter space occupation rate by splitting said space into 
"cubes" and computing the ratio of "cubical" divisions with at least one 
particle to the total (which covers all space).
"""
import numpy as np, matplotlib.pyplot as plt
import random

dim = 2
N_particles = 100

def cell_dict(ncells):
    mdim = np.array([ncells for d in range(dim)])
    m = np.zeros(mdim)
    return m

def space_occupation(points,side,thr=0.1,inc=1,matrix_2d=False):
    ncells = side**dim
    M = cell_dict(ncells)
    cell_width = 1/side
    for point in points:
        cell = tuple(int(point[i]//cell_width) for i in range(dim))
        M[cell] = 1
    occ = np.count_nonzero(M)
    r = occ/ncells
    if dim==2:
        grid_and_scatter(points, side)
        if matrix_2d:
            # Have the matrix match the 2d plot for ease of comparison
            #(x-y vs. col-row, where x and row are considered dim=0 for each 
            #case - so transpose and flip rows/x axis).
            rearranged_matrix = np.flip(M.T,axis=0) 
            print(rearranged_matrix)
    #print(r)
    if occ<=thr*N_particles: # At least 10% of particles in different cells.
        # Fine grain cells further to get a better estimate (occupation ratio 
        #will tend to be smaller).
        space_occupation(points,side+inc)
    
    return r
        
def grid_and_scatter(points, side):
    fig, axs = plt.subplots(1,figsize=(8,8))
    axs.scatter([point[0] for point in points],[point[1] for point in points],
                marker='o',s=5)
    plt.xlim((0,1))
    plt.ylim((0,1))
    
    #cell_width = 1/ncells
    hlines = [i/side for i in range(1,side)]
    vlines = hlines
    
    [axs.axhline(y=hline, color='r',linewidth=0.75,linestyle="dashed") 
             for hline in hlines] 
    [axs.axvline(x=vline, color='r',linewidth=0.75,linestyle="dashed") 
             for vline in vlines] 

min,max=0,0.5
samples = [np.array([random.uniform(min,max) for i in range(dim)]) 
           for j in range(N_particles)]
side = 2
space_occupation(samples,side)