# -*- coding: utf-8 -*-
"""
Small script used to produce a representative image for the 6-dimensional 
gaussian sampling example. A plot is generated with the starting point of the 
chain and the target density center which are used in the file 
'6dgaussian_MH_HMC'.
"""
import matplotlib.pyplot as plt

dim=6
start = [0,0,0,0,0,0]
target_center = [10,10,10,-10,-10,-10]
path = ([start,target_center])
fig, axs = plt.subplots(3,figsize=(8,8))
fig.subplots_adjust(hspace=0.35)

l = len(path)
colors = []
for i in range(l):
    colors.append((i/(l-1),0,1-i/(l-1)))

labels = ["starting point", "target distribution center"]

for j in range(dim//2):
    for i, point in enumerate(path):
        axs[j].scatter(point[j*2], point[j*2+1], marker='x',
                    s=50,color=colors[i],label=labels[i])
        axs[j].set_xlabel(r"$x_%d$" % (2*j+1))
        axs[j].set_ylabel(r"$x_%d$" % (2*j+2))
        
    xs = [path[k][j*2] for k in range(2)]
    ys = [path[k][j*2+1] for k in range(2)]
    #points = [(path[k][j*2],path[k][j*2+1]) for k in range(2)]
    axs[j].plot(xs,ys,'--',color='gray',linewidth=1)
    axs[j].legend()
    

    
