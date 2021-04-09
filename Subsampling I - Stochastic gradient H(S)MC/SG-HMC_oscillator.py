# -*- coding: utf-8 -*-
"""
Implementation of the Metropolis-Hastings and Hamiltonian Monte Carlo 
algorithms for sampling from the density given by a Rosenbrock function.
"""
import matplotlib.pyplot as plt
from autograd import grad, numpy as np

dim = 1
accepted = 0
total = 0
start = [10.]


def target_U(x):
    U = 0.5*x**2
    return(U)

def U_gradient(point, stochastic, stoch_var=4,autograd=False):
    if autograd:
        DU_f = grad(target_U)
        DU = DU_f(point)
    else:
        DU = point
    if stochastic:
        DU += np.random.normal(0,scale=stoch_var**0.5)
    return(DU)

 
def simulate_dynamics(noisy=False,initial_point=np.array(start), L=500, eta=0.1): 
    path = []   
    initial_momentum = np.array(np.random.normal(0,scale=1))
    path.append((initial_point,initial_momentum))
    
    new_point = initial_point
    DU = U_gradient(new_point,noisy)
    new_momentum = np.add(initial_momentum,-0.5*eta*DU)
    for l in range(L):
        new_point = np.add(new_point,eta*new_momentum)
        path.append((new_point,new_momentum)) # Leave out firts and last points
        #(not in "trajectory" yet and mismatched steps resp.)
        if (l != L-1):
            DU = U_gradient(new_point,noisy)
            new_momentum = np.add(new_momentum,-eta*DU)   
            
    DU = U_gradient(new_point,noisy)
    new_momentum = np.add(new_momentum,-0.5*eta*DU)
    path.append((new_point,new_momentum)) 
    return path

def momentum_step(momentum,DU,B,C,eta,M_inv):
    momentum_components = [momentum]
    momentum_components.append(-eta*DU)
    momentum_components.append(np.dot(-eta*C*M_inv,momentum))
    momentum_components.append(np.random.normal(0,scale=(2*(C-B)*eta)**0.5))
    new_momentum = sum(momentum_components)   
    return new_momentum

def simulate_dynamics_friction_leapfrog(noisy=True,V_est=4,C=3,
                                        initial_point=np.array(start),L=200, 
                                        eta=0.1, M=np.identity(dim)): 
    B = 0.5*eta*V_est
    C = B#5*B
    M_inv = M
    path = []   
    initial_momentum = np.array(np.random.normal(0,scale=1))
    path.append((initial_point,initial_momentum))
    new_point = initial_point
    
    DU = U_gradient(new_point,noisy)
    new_momentum = momentum_step(initial_momentum,DU,B,C,0.5*eta,M_inv)     
    for l in range(L):
        new_point = np.add(new_point,np.dot(eta*M_inv,new_momentum))     
        path.append((new_point,new_momentum)) 
        if (l != L-1):
            DU = U_gradient(new_point,noisy)
            new_momentum = momentum_step(new_momentum,DU,B,C,eta,M_inv)       
            
    DU = U_gradient(new_point,noisy)
    new_momentum = momentum_step(new_momentum,DU,B,C,0.5*eta,M_inv)       
    path.append((new_point,new_momentum)) 
    return path

def simulate_dynamics_friction(noisy=True,V_est=2,C=3,
                               initial_point=np.array(start), 
                               L=200, eta=0.1, M=np.identity(dim)): 
    B = 0.5*eta*V_est
    C = 5*B
    M_inv = M
    path = []   
    initial_momentum = np.array(np.random.normal(0,scale=1))
    path.append((initial_point,initial_momentum))
    new_momentum = initial_momentum
    new_point = initial_point

    for l in range(L):
        new_point = np.add(new_point,np.dot(eta*M_inv,new_momentum))
        DU = U_gradient(new_point,noisy)
        new_momentum = momentum_step(new_momentum,DU,B,C,eta,M_inv)           
        path.append((new_point,new_momentum)) 
    return path

def colored_scatter(paths, labels, markers, colors):
    fig, ax = plt.subplots(figsize=(16,16))
    #ax.set_xlim([-40,40])
    #ax.set_ylim([-40,40])
    ax.set_title("Harmonic oscillator",
                  fontsize=20,pad=20)

    for i,path in reversed(list(enumerate(paths))):
        # Assume first elements are wanted more visible and plot them last.
        if colors[i]=="gradate":
            l = len(path)
            color = []
            for j in range(l):
                color.append((j/(l-1),0,1-j/(l-1)))
        else: 
            color=colors[i]
        x,p = [x for x,p in path], [p for x,p in path]
        ax.scatter(x, p, 
                   s=4**2,marker=markers[i], color=color,label=labels[i])

        ax.set_xlabel(r"position ($\theta$)",fontsize=14)
        ax.set_ylabel("momentum ($p$)",fontsize=14)
        
    ax.legend(loc='lower right',fontsize=14)
    
def hamiltonian_dynamics():    
    which = None
    
    if which is None:
        # Plot all 3.
        p1,l1 = simulate_dynamics(noisy=False,L=1500),\
                "Correct Hamiltonian dynamics"
        p2,l2 = simulate_dynamics_friction_leapfrog(L=1500), \
                "Noisy Hamiltonian dynamics with friction"
        p3,l3 = simulate_dynamics(noisy=True,L=1500),"Noisy Hamiltonian dynamics"
        
        paths, labels = [p1,p2,p3], [l1,l2,l3]
        markers, colors = ['o','x','d'], ["black","lightgreen","gradate"]
        colored_scatter(paths,labels,markers,colors)
    else:
        # Plot only i-th path
        i=which
        colored_scatter([paths[i]],[labels[i]],[markers[i]],["gradate"])
    
def SGHD_alone():
    p,l = simulate_dynamics_friction(L=1500), \
            "Noisy Hamiltonian dynamics with friction"
    colored_scatter([p],[l],['x'],["gradate"])
    
def SGHD_leapfrog_vs_fullstep():
    p1,l1 = simulate_dynamics_friction(L=2500), \
            "Noisy Hamiltonian dynamics with friction (full step integration)"
    p2,l2 = simulate_dynamics_friction_leapfrog(L=2500), \
            "Noisy Hamiltonian dynamics with friction (leapfrog integration)"
    colored_scatter([p1,p2],[l1,l2],['x','o'],["red","black"])
    
hamiltonian_dynamics()