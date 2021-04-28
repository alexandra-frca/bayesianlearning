# -*- coding: utf-8 -*-
"""
Implementations of the Hamiltonian Monte Carlo MCMC algorithm for sampling from 
the density given by a Rosenbrock function.

Several methods for picking a sample from the generated trajectory are used:
- Simply choosing the last state, and performing a Metropolis-Hastings 
correction;
- Uniform progressive sampling, each state being chosen with probability
proportional to the target (augmented) likelihood of the corresponding point in
phase space. In this case, the trajectory is doubled until it reaches the 
desired length, so the "progressive sampling" means we're keeping one sample
position coordinate for each half trajectory (plus its cumulative - again 
augmented - likelihood);
- Partly biased progressive sampling, similar to above but biasing the samples
away from the initial point when doubling the trajectory. This bias is 
cancelled out when averaging over all possible starting points (because some 
would belong to the opposite trajectory);
- Fully biased progressive sampling, similar to above but the bias is also 
introduced when building the trajectory to be appended (i.e. when constructing
the subtree whose size should match that of the previous iteration's binary 
tree, so as to double the leaves/trajectory states).

In the last 3 strategies, the trajectory length is chosen to increase 
multiplicatively (it's doubled at each iteration).

Based on "A Conceptual Introduction to Hamiltonian Monte Carlo"
[https://arxiv.org/pdf/1701.02434.pdf]
"""
import random, matplotlib.pyplot as plt
from autograd import grad, numpy as np

dim = 2
    
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

def energy(point,momentum):
    '''
    Evaluates the "energy" at some phase space location.

    Parameters
    ----------
    point : [float]
        The position coordinates.
    momentum : [float]
        The momentum coordinates.

    Returns
    -------
    E: The "energy".

    '''
    V = target_U(point)
    K = np.sum(momentum**2)/2
    return (V+K)
    
def augmented_target(point,momentum):
    '''
    Evaluates the augmented density function at some phase space location.

    Parameters
    ----------
    point : [float]
        The position coordinates.
    momentum : [float]
        The momentum coordinates.

    Returns
    -------
    pi: The augmented density function.

    '''
    H = energy(point,momentum)
    pi = np.exp(-H)
    return(pi)

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

def leapfrog_step(point, momentum, eta):
    '''
    Approximates the system's evolution according to the Hamilton's equations 
    using leapfrog integration.

    Parameters
    ----------
    point : [float]
        The position coordinates of the starting point.
    momentum : [float]
        The momentum of the starting point.
    eta : float
        The time-step for the integration.

    Returns
    -------
    point : [float]
        The position coordinates after the integration step.
    momentum : [float]
        The momentum after the integration step.

    '''     
    DU = U_gradient(point)
    momentum = np.add(momentum,-0.5*eta*DU)
    point = np.add(point,eta*momentum) 
    DU = U_gradient(point)
    momentum = np.add(momentum,-0.5*eta*DU)
    return point, momentum
    
def build_tree(point,momentum,dir,tree_height,eta,bias):
    '''
    Constructs a trajectory in phase space by simulating the Hamiltonian 
    dynamics. 
    The total trajectory length is by construction a power of two, hence the 
    binary tree (a multiplicative expansion scheme is used where the trajectory  
    length increases by a factor of 2 at each iteration until the desired 
    base-two length is reached).

    Parameters
    ----------
    point : [float]
        The position coordinates of the extreme point of the current trajectory 
        in the direction to be considered (i.e. last/"rightmost" point if 
        dir = +1, first/"leftmost" if dir = +1).
    momentum : [float]
        The momentum coordinates of the extreme point of the current trajectory 
        in the direction to be considered (i.e. last/"rightmost" point if 
        dir = +1, first/"leftmost" if dir = +1).
    dir : int 
        (-1|1)
        The direction of time for the integration: +1 for forward time 
        evolution, -1 for backward time evolution.
    tree_height : int
        Height of the tree to be built. Single leaf trees are assigned 0 
        height.
    eta : float
        The time-step for the integration.
    bias : bool
        Whether to bias the samples in favour of the outermost states (longer
        time evolution).

    Returns
    -------
    [float]
        The position coordinates of the "leftmost" point in the trajectory 
        (earliest in time/longest backward evolution).
    [float]
        The momentum coordinates of the "leftmost" point in the trajectory 
        (earliest in time/longest backward evolution).
    [float]
        The position coordinates of the "rightmost" point in the trajectory 
        (latest in time/longest forward evolution).
    [float]
        The momentum coordinates of the "rightmost" point in the trajectory 
        (latest in time/longest forward evolution).
    active_sample : [float]
        The "active sample" of the trajectory - a position vector sampled from 
        the full trajectory corresponding to the built tree. It will 
        "represent" the subtree when joining trees.
    new_weight : float
        The weight of the full trajectory (sum of the target extended density
        evaluated at all phase space points that constitute it).

    '''
    if tree_height == 0:
        # Terminating case.
        active_sample,momentum = leapfrog_step(point,momentum,dir*eta)
        new_weight = augmented_target(active_sample,momentum)
        # We want to return leftmost AND rightmost (q,p) coordinates AND the 
        #sample (which is the only possible q here), hence all the repetition.
        return active_sample,momentum,active_sample,momentum,active_sample,\
            new_weight
    else:
        # Recursive case.
        # Construct most interior subtree (closer to initial point).
        lq,lp,rq,rp,sample0,weight0 = \
            build_tree(point,momentum,dir,tree_height-1,eta,bias)
        # Construct outermost subtree.
        if dir==-1:
            lq,lp,_,_,sample1,weight1 = \
                build_tree(lq,lp,dir,tree_height-1,eta,bias)
        if dir==1:
            _,_,rq,rp,sample1,weight1 = \
                build_tree(rq,rp,dir,tree_height-1,eta,bias)
        if bias is False:
            # Uniform sampling.
            p_out = weight1/(weight1+weight0)
        else:
            # Higher probability for the exterior tree points.
            p_out = weight1/weight0
        if random.uniform(0,1) < p_out:
            active_sample = sample1
        else:
            active_sample = sample0
        new_weight = weight0 + weight1
        return lq,lp,rq,rp,active_sample,new_weight

def uniform_progressive_sampling(initial_momentum, initial_point, L_exp, eta,
                                 biased=None):  
    '''
    Gets a HMC sample. This sample is "propagated" as the trajectory is built
    (as opposed to being picked in the end, which is more memory-intensive and
    doesn't allow for biasing). 
    
    The trajectory length is doubled at each iteration, by constructing a path
    as long as the current one (starting at its extreme) and then joining them
    together.
    
    The progressive sampling is achieved by randomly integrating forward or 
    backward in time (which picks a trajectory at random and uniformly from all 
    containing the initial point) and keeping samples of each sub-trajectory
    along with their total weights (which is enough information to get a sample 
    that is formally equivalent to one from the whole concatenated trajectory; 
    this amounts to sampling from the compound trajectory).

    Parameters
    ----------
    initial momentum : [float]
        The momentum coordinates of the initial point.
    initial_point : [float]
        The position coordinates of the initial point.
    L_exp : int, optional
        The exponent number of integration time-steps (total will be 2**L_exp). 
        The default is 6.
    eta : float
        The time-step for the integration.
    biased : str, optional
        ("doubling"|"both")
        Whether/where to bias the samples in favour of the outermost states 
        (longer time evolution). Can be never, only when doubling the tree 
        length, or also when building the new subtree to be appended. Default 
        is None (no bias).

    Returns
    -------
    active_sample : [float]
        The HMC sample.

    '''
    active_sample = initial_point
    old_weight = augmented_target(initial_point,initial_momentum)
    left_q, left_p  = initial_point, initial_momentum
    right_q, right_p  = initial_point, initial_momentum
    
    # Build subtrees to be concatenated, effectively doubling the path 
    #length L_exp times.
    for tree_height in range(L_exp):
        # Subtrees won't be biased if bias is None or 'doubling' only.
        bias_subtrees = True if biased=="both" else False
        dir = (-1)**random.randint(0,1) # -1 is backwards in time, +1 forward.
        if dir == -1: 
            # Integrate backwards in time.
            left_q, left_p, _, _, new_sample, new_weight = \
                build_tree(left_q,left_p,dir,tree_height,eta,bias_subtrees)
        if dir == 1: 
            # Integrate forward in time.
            _, _, right_q, right_p, new_sample, new_weight = \
                build_tree(right_q,right_p,dir,tree_height,eta,bias_subtrees)
        if biased is None:
            p_new = new_weight/(new_weight+old_weight)
        else:
            # Bias samples away from old trajectory/initial point.
            p_new = new_weight/old_weight
        if random.uniform(0,1) < p_new:
            active_sample = new_sample
        old_weight += new_weight
    return active_sample

first_hamiltonian_MC_step = True
def hamiltonian_MC_step(point, trajectory_sample, L_exp=6, eta=0.1):
    '''
    Performs a HMC transition.

    Parameters
    ----------
    point : [float]
        A Markov chain state.
    trajectory_sample: str 
        (last|uniform|partly biased|fully biased)
        The approach to take when obtaining a sample from the generated 
        trajectory.
    L_exp : int, optional
        The exponent number of integration time-steps (total will be 2**L_exp). 
        The default is 6.
    eta : float, optional
        The time-step. The default is 0.1.
        
    Returns
    -------
    point/new_point : [float]
        The next Markov chain state.

    '''
    global first_hamiltonian_MC_step, accepted, total
    if first_hamiltonian_MC_step:
        print("> HMC: M=I, L=2^%d, eta=%.2f | trajectory sampling: %s" % 
              (L_exp,eta,trajectory_sample))
        first_hamiltonian_MC_step = False

    initial_momentum = np.random.multivariate_normal([0,0], np.identity(dim))
    if trajectory_sample=="last":
        L = 2**L_exp
        new_point, p = simulate_dynamics(initial_momentum,point,L,eta)
        total += 1
        a=min(1,p)
        if (np.random.rand() < a):
            accepted += 1
            return(new_point)
        else:
            return(point)
    elif trajectory_sample=="uniform":
        new_point = \
            uniform_progressive_sampling(initial_momentum,point,L_exp,eta)
        return(new_point) # Correctness ensured by the sampling scheme, so p=1.
    elif trajectory_sample=="partly biased":
        new_point = \
            uniform_progressive_sampling(initial_momentum,point,L_exp,eta,
                                         biased="doubling")
        return(new_point) 
    elif trajectory_sample=="fully biased":
        new_point = \
            uniform_progressive_sampling(initial_momentum,point,L_exp,eta,
                                         biased="both")
        return(new_point) 
        
    
def hamiltonian_MC_path(steps,trajectory_sample,start=[0.,0.]):
    '''
    Constructs a HMC Markov chain.

    Parameters
    ----------
    steps : int
        The number of states the chain is to be evolved for.
    trajectory_sample: str 
        (last|uniform|partly biased|fully biased)
        The approach to take when obtaining a sample from the generated 
        trajectory.
    start : [float], optional
        The initial state. The default is [0,0].

    Returns
    -------
    path : [[float]]
        The list of ordered Markov chain states.

    '''
    path = []
    path.append(np.array(start))
    counter=0
    for t in range(1,steps+2):
        path.append(hamiltonian_MC_step(path[t-1],trajectory_sample))

        # Progress bar.
        if t==1:
            print("|0%",end="|")
        if (t%(steps/10)<1): 
            counter+=10
            print(counter,"%",sep="",end="|")
    print("") # For newline.
    if trajectory_sample=="last":
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
    
def main():
    steps=10000
    print("Number of MCMC steps: ", steps)
    
    # Last works poorly for higher L
    trajectory_sample = "last" # last|uniform|partly biased|fully biased
    HMC_path = hamiltonian_MC_path(steps,trajectory_sample)
    colored_scatter(HMC_path,title=
                        ("Hamiltonian Monte Carlo (trajectory sampling: %s)" 
                        % trajectory_sample))
main()