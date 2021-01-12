# -*- coding: utf-8 -*-
"""
Script to generate a set of points distributed according to a multimodal  
distribution making the shape of a smiley face, plot them, and save them 
in a file as a list of cumulative datachunks.

A fixed amount of points is attributed to the the regions around each mode (3 in
total: the 2 eyes, and the mouth), and they are obtained using Hamiltonian Monte 
Carlo chains on the local density (discarding an appropriate amount of burn-in
and lag samples so as to better cover these densities).
"""
import sys, pickle, matplotlib.pyplot as plt
from autograd import grad, numpy as np
from itertools import accumulate

dim = 2
accepted = 0
total = 0

initial_point=[0.,13.]

def multivariate_gaussian(x, mu, sigma, normalize=False):
    x_mu = np.array(x-mu)
    #x_mu_t = np.transpose(x_mu)
    sigma_inv = np.linalg.inv(sigma)
    power = -0.5*np.linalg.multi_dot([x_mu,sigma_inv,x_mu])

    k = len(x)
    sigma_det = np.linalg.det(sigma)
    norm = ((2*np.pi)**k*sigma_det)**0.5
    if normalize:
        return(np.exp(power))
    else:
        return(np.exp(power)/norm)
    
def target(x,i):
    g = []
    g.append(np.exp(1/5*(-6*(-(2.5-x[0])**2-1.5*x[1]+38)**2-(2.5-x[0])**2)))
    g.append(np.exp(1/5*(-6*(-(2.5+x[0])**2-1.5*x[1]+38)**2-(2.5+x[0])**2)))
    g.append(np.exp(1/5*(-5*(x[1]-x[0]**2)**2-x[0]**2)))
    return (g[i])

def target_U(x,i):
    return(-np.log(target(x,i)))

def target_DU(x,i):
    dg = []
    g1 = np.exp(1/5*(-6*(-(2.5-x[0])**2-1.5*x[1]+38)**2-(2.5-x[0])**2))
    dg1dx = 0.2*(-12*(5-2*x[0])*(-(2.5-x[0])**2-1.5*x[1]+38)+(5-2*x[0]))*g1
    dg1dy = 0.2*18*(-(2.5-x[0])**2-1.5*x[1]+38)*g1
    dg.append(np.array([dg1dx,dg1dy]))
    
    g2 = np.exp(1/5*(-6*(-(2.5+x[0])**2-1.5*x[1]+38)**2-(2.5+x[0])**2))
    dg2dx = 0.2*(+12*(5+2*x[0])*(-(2.5+x[0])**2-1.5*x[1]+38)-(5+2*x[0]))*g2
    dg2dy = 0.2*18*(-(2.5+x[0])**2-1.5*x[1]+38)*g2
    dg.append(np.array([dg2dx,dg2dy]))
    
    g3 = np.exp(1/5*(-5*(x[1]-x[0]**2)**2-x[0]**2))
    dg3dx = 0.2*(20*x[0]*(x[1]-x[0]**2)-2*x[0])*g3
    dg3dy = -2*(x[1]-x[0]**2)*g3
    dg.append(np.array([dg3dx,dg3dy]))
    return(dg[i])

def U_gradient(point,i):
    DU = target_DU(point,i)
    return(DU)

def metropolis_hastings_step(point, sigma=0.05):
    global dim, accepted, total
    Sigma = sigma*np.identity(dim)
    new_point = np.random.multivariate_normal(point, Sigma)

    p = target(new_point)*multivariate_gaussian(point,new_point,Sigma)/ \
        target(point)*multivariate_gaussian(new_point,point,Sigma)
    a = min(1,p)

    total += 1
    if (np.random.rand() < a):
        accepted += 1
        return(new_point)
    else:
        return(point)

def metropolis_hastings_path(steps,start=initial_point):
    global accepted, total
    path = []
    path.append(np.array(start))
    for t in range(1,steps+1):
        path.append(metropolis_hastings_step(path[t-1]))
    print("MH: %d%% particle acceptance rate. " % (100*accepted/total))
    accepted, total = 0, 0
    return path

def simulate_dynamics(initial_momentum, initial_point, i, L, eta):        
    new_point = initial_point
    DU = U_gradient(new_point,i)
    new_momentum = np.add(initial_momentum,-0.5*eta*DU)
    for l in range(L):
        new_point = np.add(new_point,eta*new_momentum)
        if (l != L-1):
            DU = U_gradient(new_point,i)
            new_momentum = np.add(new_momentum,-eta*DU)     
    new_momentum = np.add(new_momentum,-0.5*eta*DU)
    new_momentum = -new_momentum 
    
    p = np.exp(target_U(initial_point,i)-target_U(new_point,i)+
            np.sum(initial_momentum**2)/2-np.sum(new_momentum**2)/2)
    return new_point, p
        
def hamiltonian_MC_step(point, i, M=np.identity(dim), L=20, eta=0.005):
    global accepted, total
    
    #if (i>=2):
     #   L=15
        
    initial_momentum = np.random.multivariate_normal([0,0], M)
    new_point, p = simulate_dynamics(initial_momentum,point,i,L,eta)
    
    total += 1
    a=min(1,p)
    if (np.random.rand() < a):
        accepted += 1
        return(new_point)
    else:
        return(point)
    
def hamiltonian_MC_path(points,i,burn_in=10,lag=5,start=initial_point):   
    if (i>=2):
        lag=20
    path = []
    path.append(np.array(start))
    for b in range(burn_in):
        path[0]=hamiltonian_MC_step(path[0],i)
    for t in range(1,points):
        path.append(hamiltonian_MC_step(path[t-1],i))
        for l in range(lag):
            path[t]=hamiltonian_MC_step(path[t],i)
    print("HMC: %d%% particle acceptance rate. " % (100*accepted/total))
    return path

def parallel_HMC(n_particles,steps,j):
    means = np.array([0,10])
    Sigma = np.matrix([[10,0],[0,20]])
    particles = np.random.multivariate_normal(means,Sigma,n_particles)
    for t in range(steps):
        for i,particle in enumerate(particles):
            particles[i]=hamiltonian_MC_step(particle,j)
        sys.stdout.write('.');sys.stdout.flush();
    print("\nParallel HMC: %d%% particle acceptance rate. " % (100*accepted/total))
    return particles

def simple_scatter(points):
    fig, axs = plt.subplots(1,figsize=(12,8))
    axs.scatter([point[0] for point in points],[point[1] for point in points],
                marker='o',s=8)
    plt.xlim((-6,6))
    plt.ylim((-3,30))
    
def colored_scatter(path):
    fig, axs = plt.subplots(3,figsize=(8,8))
    fig.subplots_adjust(hspace=0.35)
    
    l = len(path)
    colors = []
    for i in range(l):
        colors.append((i/(l-1),0,1-i/(l-1)))

    for i, point in enumerate(path):
        for j in range(dim//2):
            axs[j].scatter(point[j*2], point[j*2+1], marker='o',
                        s=5,color=colors[i])
            axs[j].set_xlabel(r"$x_%d$" % (2*j+1))
            axs[j].set_ylabel(r"$x_%d$" % (2*j+2))
      
def colored_scatters(paths, titles=None):
    fig, axs = plt.subplots(len(paths),1,figsize=(8,8))
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.3)
    
    for n,path in enumerate(paths):
        l = len(path)
        colors = []
        for i in range(l):
            colors.append((i/(l-1),0,1-i/(l-1)))
        
        for i, point in enumerate(path):
            axs[n].scatter(point[0], point[1], marker='o',
                        s=5,color=colors[i])
            axs[n].set_xlabel("x")
            axs[n].set_ylabel("y")
                
        if (titles!=None):
            axs[n].set_title(titles[n],fontsize=14,pad=15)
    
def shuffle_split_accumulate(arr,chunksize=100):
    np.random.shuffle(arr)
    arr = [arr[i:i+chunksize] for i in range(0, len(arr), chunksize)]
    # The last datachunk will have len(arr)%100 points.
    arr = list(accumulate(arr))
    return(arr)
    
def main():
    starts=[]
    starts.append(np.array([2.,25.])) # Right eye.
    starts.append(np.array([-2,25.])) # Left eye.
    starts.append(np.array([0.,1.]))  # Mouth.

    points=[]
    for i in range(3):
        points.append(hamiltonian_MC_path(256,i,start=starts[i]))
    points=points[0]+points[1]+points[2]

    points = shuffle_split_accumulate(points)
    simple_scatter(points[len(points)-1])

    with open('smileyface3.data', 'wb') as filehandle:
        pickle.dump(points, filehandle)
    
main()