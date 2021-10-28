# -*- coding: utf-8 -*-
"""
Implementation of the Metropolis-Hastings and Hamiltonian Monte Carlo 
algorithms for sampling from a 6-dimensional normal distribution.
"""
import matplotlib.pyplot as plt
from autograd import grad, numpy as np

dim = 6
accepted = 0
total = 0

def gaussian(x, mu, sigma, normalize=False):
    power = -(x-mu)**2/(2*sigma**2)
    norm = (2*np.pi*sigma**2)**0.5
    if not normalize:
        return np.exp(power)
    else:
        return np.exp(power)/norm

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

def multivariate_gaussian_log_gradient(x, mu, sigma):
    x_mu = np.array(x-mu)
    sigma_inv = np.linalg.inv(sigma)
    return(np.dot(sigma_inv,x_mu))
    
def target(xs):
    means = np.array([15,15,15,-15,-15,-15])
    #means = np.array([20,20,20,-20,-20,-20])
    #means = np.array([17.5,17.5,17.5,-17.5,-17.5,-17.5])
    Sigma = np.identity(dim)
    return multivariate_gaussian(xs, means, Sigma)

def target_U(xs):
    return(-np.log(target(xs)))

def target_DU(xs):
    means = np.array([15,15,15,-15,-15,-15])
    #means = np.array([20,20,20,-20,-20,-20])
    #means = np.array([17.5,17.5,17.5,-17.5,-17.5,-17.5])
    Sigma = np.identity(dim)
    return multivariate_gaussian_log_gradient(xs, means, Sigma)

first_rwm = True
def metropolis_hastings_step(point, sigma=0.2):
    global dim, accepted, total, first_rwm
    if first_rwm:
      print("RWM: sigma = %.2f" % sigma)
      first_rwm = False
      
    Sigma = sigma*np.identity(dim)
    new_point = np.random.multivariate_normal(point, Sigma)

    p = target(new_point)/target(point)
    a = min(1,p)

    total += 1
    if (np.random.rand() < a):
        accepted += 1
        return(new_point)
    else:
        return(point)

def metropolis_hastings_path(steps,start=[0,0,0,0,0,0]):
    global accepted, total
    path = []
    path.append(np.array(start))
    for t in range(1,steps+1):
        path.append(metropolis_hastings_step(path[t-1]))
    print("MH: %.1f%% particle acceptance rate. " % (100*accepted/total))
    accepted, total = 0, 0
    return path
    
def U_gradient(point,autograd=False):
    if not autograd:
        DU = target_DU(point)
    else:
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
        
def hamiltonian_MC_step(point, M=np.identity(dim), L=30, eta=0.05):
    global accepted, total
    
    initial_momentum = np.random.multivariate_normal([0,0,0,0,0,0], M)
    new_point, p = simulate_dynamics(initial_momentum,point,L,eta)
    
    total += 1
    a=min(1,p)
    if (np.random.rand() < a):
        accepted += 1
        return(new_point)
    else:
        return(point)
    
def hamiltonian_MC_path(steps,start=[0.,0.,0.,0.,0.,0.]):
    path = []
    path.append(np.array(start))
    for t in range(1,steps+1):
        path.append(hamiltonian_MC_step(path[t-1]))
    print("HMC: %.1f%% particle acceptance rate. " % (100*accepted/total))
    return path

def colored_scatter(path):
    fig, axs = plt.subplots(3,figsize=(8,8))
    #fig.subplots_adjust(hspace=0.35)
    
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
    fig, axs = plt.subplots(len(paths),3,figsize=(24,16))
    #fig.subplots_adjust(hspace=0.3)
    #fig.subplots_adjust(wspace=0.3)
    
    for n,path in enumerate(paths):
        l = len(path)
        colors = []
        for i in range(l):
            colors.append((i/(l-1),0,1-i/(l-1)))
        
        for i, point in enumerate(path):
            for j in range(dim//2):
                axs[n,j].scatter(point[j*2], point[j*2+1], marker='x' if i==0 else 'o',
                            s=100,color=colors[i])
                axs[n,j].set_xlabel(r"$x_%d$" % (2*j+1),size=20)
                axs[n,j].set_ylabel(r"$x_%d$" % (2*j+2),size=20)
                
        if (titles!=None):
          for j in range(dim//2):
            ttl = titles[n] + (" (dimensions %d and %d)" % (2*j+1,2*j+2))
            axs[n,j].set_title(ttl,fontsize=25,pad=15)
    
def main():
    steps=1000
    MH_path = metropolis_hastings_path(steps)
    HMC_path = hamiltonian_MC_path(steps)

    colored_scatters([MH_path,HMC_path], 
                     titles=["RWM","HMC"])
    
main()