# -*- coding: utf-8 -*-
"""
Dynamic implementations of the Hamiltonian Monte Carlo MCMC algorithm for 
sampling from the density given by a Rosenbrock function.

The number of leapfrog integration steps is chosen adaptively, based on a  
termination criterion meant to stop the path short of redundant exploration
(the no-U-turn condition). Before reaching it, the trajectory length is 
increased multiplicatively (doubled at each iteration).

Multinomial sampling is used to pick samples among the trajectory states. 
Biased progressive sampling is used when doubling trajectories, and uniform
progressive sampling when creating the second half of the trajectory (to be 
appended when doubling).

Based on "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian 
Monte Carlo" 
[https://arxiv.org/pdf/1111.4246.pdf]
and "A Conceptual Introduction to Hamiltonian Monte Carlo"
[https://arxiv.org/pdf/1701.02434.pdf]
"""
import random, matplotlib.pyplot as plt
from autograd import grad, numpy as np

dim = 2
accepted = 0
total = 0
    
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
    E = V+K
    return E
    
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

total_leapfrog_steps = 0
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
    global total_leapfrog_steps
    total_leapfrog_steps+=1 
    
    DU = U_gradient(point)
    momentum = np.add(momentum,-0.5*eta*DU)
    point = np.add(point,eta*momentum) 
    DU = U_gradient(point)
    momentum = np.add(momentum,-0.5*eta*DU)
    return point, momentum
    
def build_tree(point,momentum,dir,tree_height,eta):
    '''
    Constructs a trajectory in phase space by simulating the Hamiltonian 
    dynamics. 
    The total trajectory length will be by construction a power of two, hence  
    the binary tree (a multiplicative expansion scheme is used where the   
    trajectory length increases by a factor of 2 at each iteration until the 
    termination criterion is satisfied).

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
    premature_termination : bool
        Whether the termination condition has been fulfilled during the 
        construction of the tree. A return value of True means that the tree
        must be discarded, because it undermines reversibility.

    '''
    premature_termination = False
    if tree_height == 0:
        # Terminating case.
        active_sample,momentum = leapfrog_step(point,momentum,dir*eta)
        new_weight = augmented_target(active_sample,momentum)
        # We want to return leftmost and rightmost (q,p) coordinates and the 
        #sample (which is the only possible q here), hence all the repetition.
        return active_sample,momentum,active_sample,momentum,active_sample,\
            new_weight, premature_termination
    else:
        # Recursive case.
        # Construct most interior subtree (closer to initial point).
        lq,lp,rq,rp,sample0,weight0, premature_termination = \
            build_tree(point,momentum,dir,tree_height-1,eta)
        if premature_termination:
            # Won't matter, the whole tree will be discarded anyway.
            active_sample, new_weight = sample0, weight0
        else:
            # Construct outermost subtree.
            if dir==-1:
                lq, lp, _, _, sample1, weight1, premature_termination = \
                    build_tree(lq,lp,dir,tree_height-1,eta)
            if dir==1:
                _, _, rq, rp, sample1, weight1, premature_termination = \
                    build_tree(rq,rp,dir,tree_height-1,eta)
            # We'll use uniform sampling here.
            p_out = weight1/(weight1+weight0)
            if random.uniform(0,1) < p_out:
                active_sample = sample1
            else:
                active_sample = sample0
            new_weight = weight0 + weight1
            premature_termination = (np.dot(rp,(rq - lq)) < 0) \
                or (np.dot(-lp,(lq - rq)) < 0)
        return lq,lp,rq,rp,active_sample,new_weight, premature_termination

discarded_halftrees = 0
def no_u_turn_sampler(initial_momentum, initial_point, eta):  
    '''
    Gets a HMC sample (dynamic implementation). The integration path length is 
    chosen adaptively, increasing by a factor of 2 until the extreme points 
    fulfill the no-U-turn termination condition.
    
    This sample is "propagated" as the trajectory is built (as opposed to being 
    picked in the end, which is more memory-intensive and doesn't allow for 
    biasing).
    
    The expansions are made by constructing a new path as long as the current 
    one (starting at its extreme) and then joining them together. The 
    integration is chosen at random to progress forward or backward in time 
    (which picks a trajectory at random and uniformly from all containing the 
    initial point), and samples of each sub-trajectory are kept along with 
    their total weights (which is enough information to get a sample that is 
    formally equivalent to one from the whole concatenated trajectory; this 
    amounts to sampling from the compound trajectory).
    
    If the termination criterion is satisfied by any subtree (i.e. other than
    when doubling), detailed balance would be violated by including the half
    tree that contains it: should the trajectory have started at a point in  
    that subtree, it may never reach points outside of it, whereas any starting
    point should be able to generate in full every trajectory to which it 
    belongs with the same probability as any other point in that trajectory.
    As such, in cases like these the whole new trajectory is discarded, the old 
    one is kept, the execution terminates and a sample is returned (from the 
    old trajectory).

    Parameters
    ----------
    initial momentum : [float]
        The momentum coordinates of the initial point.
    initial_point : [float]
        The position coordinates of the initial point.
    eta : float
        The time-step for the integration.

    Returns
    -------
    active_sample : [float]
        The HMC sample.

    '''
    active_sample = initial_point
    old_weight = augmented_target(initial_point,initial_momentum)
    left_q, left_p  = initial_point, initial_momentum
    right_q, right_p  = initial_point, initial_momentum
    tree_height = 0
    terminate = False 
    while not terminate:
        dir = (-1)**random.randint(0,1) # -1 is backwards in time, +1 forward.
        if dir == -1: # Integrate backwards in time.
            # Subtrees won't be biased if bias is None or 'doubling' only.
            left_q, left_p, _, _, new_sample, new_weight, \
                broke_reversibility = build_tree(left_q,left_p,
                                                 dir,tree_height,eta)
        if dir == 1:
            _, _, right_q, right_p, new_sample, new_weight, \
                broke_reversibility = build_tree(right_q,right_p,
                                                 dir,tree_height,eta)
        # Bias points away from old trajectory/initial point.
        p_new = new_weight/old_weight
        if broke_reversibility:
            global discarded_halftrees
            discarded_halftrees += 1
        # If reversibility was broken, the entire half trajectory must be 
        #discarded (hence the "else").
        elif (random.uniform(0,1) < p_new):
            active_sample = new_sample
        old_weight += new_weight
        l = (np.dot(right_p,(right_q - left_q)) < 0)
        r = (np.dot(-left_p,(left_q - right_q)) < 0)
        u_turn = l or r # Will be True when one of the trajectory ends is 
        #expected to decrease the distance to the other's last position 
        #judging by its current momentum.
        terminate = u_turn or broke_reversibility
        tree_height += 1
    return active_sample

first_hamiltonian_MC_step = True
def hamiltonian_MC_step(point, eta=0.1):
    '''
    Performs a HMC transition.

    Parameters
    ----------
    point : [float]
        A Markov chain state.
    eta : float, optional
        The time-step. The default is 0.1.

    Returns
    -------
    new_point : [float]
        The next Markov chain state.

    '''
    global first_hamiltonian_MC_step, accepted, total
    if first_hamiltonian_MC_step:
        print("> HMC: M=I, adaptive L, eta=%.2f | dynamic (no-U-turn sampler)"  
              % eta)
        first_hamiltonian_MC_step = False

    initial_momentum = np.random.multivariate_normal([0,0], np.identity(dim))

    new_point = no_u_turn_sampler(initial_momentum,point,eta)
    return(new_point) 

    
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
    counter=0
    for t in range(1,steps+2):
        path.append(hamiltonian_MC_step(path[t-1]))

        # Progress bar.
        if t==1:
            print("|0%",end="|")
        if (t%(steps/10)<1): 
            counter+=10
            print(counter,"%",sep="",end="|")
    print("") # For newline.
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
    
    traj_sample = "last" # last|uniform|partly biased|fully biased
    HMC_path = hamiltonian_MC_path(steps)
    print("Average leapfrog steps per MCMC step (total generated): 2^%.2f" %
          np.log2(total_leapfrog_steps/steps))
    print("Fraction of discarded half-trees when doubling: %d%%" %
          round(100*discarded_halftrees/steps))
    colored_scatter(HMC_path,title=
                        ("Hamiltonian Monte Carlo (trajectory sampling: %s)" 
                        % traj_sample))
main()