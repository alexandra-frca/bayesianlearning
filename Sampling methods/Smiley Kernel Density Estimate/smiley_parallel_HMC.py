# -*- coding: utf-8 -*-
"""
Sampling from a smiley kernel density estimate using a parallel Hamiltonian
Monte Carlo approach. 
"""
import sys, pickle, matplotlib.pyplot as plt
from autograd import grad, numpy as np

dim = 2
accepted = 0
total = 0
points=[]

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
    
def kernel_density_estimate(x, points):
    n = len(points)
    h = n**-0.2
    kde = 0
    for point in points:
        kde += multivariate_gaussian(x,point,h*np.identity(dim),
                                    normalize=True)/(n*h)
    return(kde)

def target_U(x):
    global points
    return(-np.log(kernel_density_estimate(x,points)))

def U_gradient(point):
    DU_f = grad(target_U)
    DU = DU_f(point)
    return(DU)

def simulate_dynamics(initial_momentum, initial_point, L, eta):        
    new_point = initial_point
    DU = U_gradient(new_point)
    new_momentum = np.add(initial_momentum,-0.5*eta*DU)
    for l in range(L):
        new_point = np.add(new_point,eta*new_momentum)
        DU = U_gradient(new_point)
        if (l != L-1):
            new_momentum = np.add(new_momentum,-eta*DU)     
    new_momentum = np.add(new_momentum,-0.5*eta*DU)
    new_momentum = -new_momentum 
    
    p = np.exp(target_U(initial_point)-target_U(new_point)+
            np.sum(initial_momentum**2)/2-np.sum(new_momentum**2)/2)
    return new_point, p
        
def hamiltonian_MC_step(point, M=np.identity(dim), L=20, eta=0.05):
    global accepted, total
    
    initial_momentum = np.random.multivariate_normal([0,0], M)
    new_point, p = simulate_dynamics(initial_momentum,point,L,eta)
    
    total += 1
    a=min(1,p)
    if (np.random.rand() < a):
        accepted += 1
        return(new_point)
    else:
        return(point)

def parallel_HMC(n_particles,steps):
    means = np.array([0,10])
    Sigma = np.matrix([[10,0],[0,20]])
    particles = np.random.multivariate_normal(means,Sigma,n_particles)
    for t in range(steps):
        for i,particle in enumerate(particles):
            particles[i]=hamiltonian_MC_step(particle)
        sys.stdout.write('.');sys.stdout.flush();
    print("\nParallel HMC: %d%% particle acceptance rate. " % (100*accepted/total))
    return particles

def simple_scatter(points):
    fig, axs = plt.subplots(1,figsize=(8,8))
    axs.scatter([point[0] for point in points],[point[1] for point in points],
                marker='o',s=5)
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
    
def main():
    with open('smileyface2.data', 'rb') as filehandle:
        data = pickle.load(filehandle)
        
    global points 
    points= data[len(data)-1] # Use the last item of the list of cumulative 
    #datachunks, which contains all data - thereby providing the final density 
    #considering all points. 
    
    steps = len(data)
    n_particles = 128

    particles = parallel_HMC(n_particles,steps)
    simple_scatter(particles)

    
main()