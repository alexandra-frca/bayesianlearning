# -*- coding: utf-8 -*-
"""
Implementation of the Metropolis-Hastings and Hamiltonian Monte Carlo 
algorithms for sampling from the density given by a Rosenbrock function.
"""
import matplotlib.pyplot as plt
from autograd import grad, numpy as np

dim = 2
accepted = 0
total = 0

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
    
def target(x):
    g = np.exp(1/8*(-5*(x[1]-x[0]**2)**2-x[0]**2))
    return g

def target_U(x):
    return(-np.log(target(x)))

def metropolis_hastings_step(point, sigma=0.2):
    global dim, accepted, total
    Sigma = sigma*np.identity(dim)
    new_point = np.random.multivariate_normal(point, Sigma)

    p = target(new_point)*multivariate_gaussian(point,new_point,Sigma)/ \
        (target(point)*multivariate_gaussian(new_point,point,Sigma))
    a = min(1,p)

    total += 1
    if (np.random.rand() < a):
        accepted += 1
        return(new_point)
    else:
        return(point)

def metropolis_hastings_path(steps,start=[0,0]):
    global accepted, total
    path = []
    path.append(np.array(start))
    for t in range(1,steps+1):
        path.append(metropolis_hastings_step(path[t-1]))
    print("MH: %d%% particle acceptance rate. " % (100*accepted/total))
    accepted, total = 0, 0
    return path
    
def U_gradient(point,autograd=True):
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
    new_momentum = -new_momentum #?
    
    p = np.exp(target_U(initial_point)-target_U(new_point)+
            np.sum(initial_momentum**2)/2-np.sum(new_momentum**2)/2)
    return new_point, p
        
def hamiltonian_MC_step(point, M=np.identity(dim), L=35, eta=0.03):
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
    
def hamiltonian_MC_path(steps,start=[0.,0.]):
    path = []
    path.append(np.array(start))
    for t in range(1,steps+1):
        path.append(hamiltonian_MC_step(path[t-1]))
    print("HMC: %d%% particle acceptance rate. " % (100*accepted/total))
    return path

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
    steps=1000
    MH_path = metropolis_hastings_path(steps)
    HMC_path = hamiltonian_MC_path(steps)

    colored_scatters([MH_path,HMC_path], 
                     titles=["Metropolis-Hastings","Hamiltonian Monte Carlo"])
    
main()