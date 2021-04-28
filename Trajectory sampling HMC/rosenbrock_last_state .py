# -*- coding: utf-8 -*-
"""
Implementation of Hamiltonian Monte Carlo and random walk Metropolis MCMC
algorithms for sampling from the density given by a Rosenbrock function.

In HMC, only the last state of the trajectory is kept, and a Metropolis 
correction is performed to guarantee correctness.
"""
import matplotlib.pyplot as plt
from autograd import grad, numpy as np

dim = 2
accepted = 0
total = 0

def multivariate_gaussian(x, mu, sigma, normalize=False):
    '''
    Evaluates a multivariate gaussian function at some point.

    Parameters
    ----------
    x : [float]
        The point.
    mu : [float]
        The mean.
    sigma : TYPE
        The ordered square roots of the diagonal of the covariance matrix 
        (standard deviations along each dimension). 
    normalize : bool, optional
        Whether to normalize the result. The default is False.

    Returns
    -------
    e: float
        The result.
    '''
    x_mu = np.array(x-mu)
    #x_mu_t = np.transpose(x_mu)
    sigma_inv = np.linalg.inv(sigma)
    power = -0.5*np.linalg.multi_dot([x_mu,sigma_inv,x_mu])

    k = len(x)
    sigma_det = np.linalg.det(sigma)
    norm = ((2*np.pi)**k*sigma_det)**0.5
    e = np.exp(power)
    if normalize:
        return(e)
    else:
        e = e/norm
        return(e)
    
def target(x):
    '''
    Evaluates the target likelihood at a point.

    Parameters
    ----------
    x : [float]
        The point.

    Returns
    -------
    g : float
        The target likelihood.

    '''
    g = np.exp(1/8*(-5*(x[1]-x[0]**2)**2-x[0]**2))
    return g

def target_U(x):
    '''
    Evaluates the target "potential energy" at a point (negated loglikelihood).

    Parameters
    ----------
    x : [float]
        The point.

    Returns
    -------
    U: The "potential".

    '''
    U = -np.log(target(x))
    return(U)

first_metropolis_hastings_step = True
def metropolis_hastings_step(point, sigma=0.15):
    '''
    Performs a random walk Metropolis transition.

    Parameters
    ----------
    point : [float]
        A Markov chain state.
    sigma : float, optional
        The variancefor the proposals. It will be used to populate the diagonal 
        of the covariance matrix. The default is 0.15.

    Returns
    -------
    point: The next Markov chain state.

    '''
    global first_metropolis_hastings_step
    if first_metropolis_hastings_step:
        print("> Random walk Metropolis: sigma = diag(%.2f)" % sigma)
        first_metropolis_hastings_step = False
        
    global dim, accepted, total
    Sigma = sigma*np.identity(dim)
    new_point = np.random.multivariate_normal(point, Sigma)
    p = target(new_point)/target(point)
    
    a = min(1,p)
    total += 1
    if (np.random.rand() < a):
        accepted += 1
        point = new_point
        return(point)
    else:
        return(point)

def metropolis_hastings_path(steps,start=[0,0]):
    '''
    Constructs a random walk Metropolis Markov chain.

    Parameters
    ----------
    steps : int
        The number of states the chain is to be evolved for.
    start : [float], optional
        The initial state. The default is [0,0].

    Returns
    -------
    path : [[float]]
        The list of ordered Markov chain states.

    '''
    global accepted, total
    path = []
    path.append(np.array(start))
    for t in range(1,steps+1):
        path.append(metropolis_hastings_step(path[t-1]))
    print("MH: %d%% particle acceptance rate. " % (100*accepted/total))
    accepted, total = 0, 0
    return path
    
def U_gradient(point,autograd=False):
    '''
    Evaluates the gradient of the target "potential energy" at a point (negated 
    log-gradient).

    Parameters
    ----------
    point : [float]
        The point.
    autograd : bool, optional
        Whether to use automatic differentiation. The default is False.

    Returns
    -------
    DU: [float]
        The evaluated gradient.
    '''
    if autograd:
        DU_f = grad(target_U)
        DU = DU_f(point)
    else:
        x,y = point
        dUdx = -1/8*(-10*(-2*x)*(y-x**2)-2*x)
        dUdy = -1/8*(-10*(y-x**2))
        DU = np.array([dUdx,dUdy])
    return(DU)

def simulate_dynamics(initial_momentum, initial_point, L, eta):   
    '''
    Approximates the system's evolution according to the Hamilton's equations 
    using leapfrog integration.

    Parameters
    ----------
    initial_momentum : [float]
        The momentum of the starting point.
    initial_point : [float]
        The position coordinates of the starting point.
    L : int
        The number of integration time-steps.
    eta : float
        The time-step.

    Returns
    -------
    new_point : [float]
        The position coordinates of the ending point.
    p : [float]
        The momentum of the ending point.

    '''     
    new_point = initial_point
    DU = U_gradient(new_point)
    new_momentum = np.add(initial_momentum,-0.5*eta*DU)
    for l in range(L):
        new_point = np.add(new_point,eta*new_momentum)
        DU = U_gradient(new_point)
        if (l != L-1):
            new_momentum = np.add(new_momentum,-eta*DU)     
    new_momentum = np.add(new_momentum,-0.5*eta*DU)
    
    p = np.exp(target_U(initial_point)-target_U(new_point)+
            np.sum(initial_momentum**2)/2-np.sum(new_momentum**2)/2)
    return new_point, p
        
first_hamiltonian_MC_step = True
def hamiltonian_MC_step(point, L=40, eta=0.1):
    '''
    Performs a HMC transition.

    Parameters
    ----------
    point : [float]
        A Markov chain state.
    L : int, optional
        The number of integration time-steps. The default is 40.
    eta : float, optional
        The time-step. The default is 0.1.

    Returns
    -------
    point : [float]
        The next Markov chain state.

    '''
    global first_hamiltonian_MC_step, accepted, total
    if first_hamiltonian_MC_step:
        print("> HMC: M=I, L=%d, eta=%.10f" % (L,eta))
        first_hamiltonian_MC_step = False
        
    M=np.identity(dim)
    initial_momentum = np.random.multivariate_normal([0,0], M)
    new_point, p = simulate_dynamics(initial_momentum,point,L,eta)
    
    total += 1
    a=min(1,p)
    if (np.random.rand() < a):
        accepted += 1
        point = new_point
        return(point)
    else:
        return(point)
    
def hamiltonian_MC_path(steps,start=[0.,0.]):
    '''
    Constructs a HMC Markov chain.

    Parameters
    ----------
    steps : int
        The number of states the chain is to be evolved for.
    start : [float], optional
        The initial state. The default is [0,0].

    Returns
    -------
    path : [[float]]
        The list of ordered Markov chain states.

    '''
    path = []
    path.append(np.array(start))
    for t in range(1,steps+1):
        path.append(hamiltonian_MC_step(path[t-1]))
    print("HMC: %d%% particle acceptance rate. " % (100*accepted/total))
    return path

def colored_scatter(path, title=None):
    '''
    Plots a sequence of points with a color gradient (blue to red).

    Parameters
    ----------
    path : [(float,float)]
        An ordered list of coordinate pairs.
    title : str, optional
        A title for the plot. The default is None.

    Returns
    -------
    None.

    '''
    fig, axs = plt.subplots(1,figsize=(12,12))
    fig.subplots_adjust(hspace=0.35)
    axs.set_xlim((-8,8))
    axs.set_ylim((-5,65))
    
    l = len(path)
    colors = []
    for i in range(l):
        colors.append((i/(l-1),0,1-i/(l-1)))
    
    xs,ys = [point[0] for point in path], [point[1] for point in path]
    axs.scatter(xs, ys, marker='o',s=5,color=colors)
        
    axs.set_xlabel("x")
    axs.set_ylabel("y")
    if title is not None:
        axs.set_title(title, fontsize=20)
      
def colored_scatters(paths, titles=None):
    '''
    Plots 2 sequences of points with a color gradient (blue to red) in 
    vertically aligned subplots.

    Parameters
    ----------
    paths : ([(float,float)],[(float,float)])
        An ordered list of coordinate pairs.
    title : [str], optional
        Titles for the plots, given by the same order by which their paths are 
        listed. The default is None.

    Returns
    -------
    None.

    '''
    fig, axs = plt.subplots(len(paths),1,figsize=(12,16))
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
    which = "both" # HMC | RWM | both
    steps=10000
    if which=="HMC":
        HMC_path = hamiltonian_MC_path(steps)
        colored_scatter(HMC_path,title="Hamiltonian Monte Carlo")
    elif which=="RWM":
        MH_path = metropolis_hastings_path(steps)
        colored_scatter(MH_path,title="Random Walk Metropolis")
    elif which=="both":
        HMC_path = hamiltonian_MC_path(steps)
        MH_path = metropolis_hastings_path(steps)
        colored_scatters([MH_path,HMC_path], 
                titles=["Random Walk Metropolis","Hamiltonian Monte Carlo"])

main()