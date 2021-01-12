# -*- coding: utf-8 -*-
"""
Sampling from a smiley kernel density estimate using a sequential Hamiltonian
Monte Carlo approach, using a leave-one-out kernel density estimate of the 
latest available SMC distribution approximation instead of the previous step's 
theoretical distribution in the denominator when updating the weights (in the 
correction phase).
"""
import sys, random, pickle, matplotlib.pyplot as plt
from autograd import grad, numpy as np

curr_data = []
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
    
def kernel_density_estimate(x, points, leave_out=None):
    global a
    if leave_out is None:
        n = len(points)
    else:
        n = len(points)-1
    h = n**-0.2
    kde = 0
    for point in points:
        if not np.array_equal(point,leave_out):
            kde += multivariate_gaussian(x,point,h*np.identity(dim),
                                    normalize=True)/(n*h)
    return(kde)

def target(x,points):
    return(kernel_density_estimate(x, points))

def target_U(x):
    global curr_data
    return(-np.log(kernel_density_estimate(x,curr_data)))

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
        if (l != L-1):
            DU = U_gradient(new_point)
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
    
def selection_and_mutation(particles):
    n = len(particles)
    selected_particles = random.choices(list(particles.keys()), 
                                      weights=particles.values(), k=n)
    particles.clear()
    for key in selected_particles:
        particle = np.frombuffer(key,dtype='float64')
        repeated = True
        while (repeated == True):
            mutated_particle = hamiltonian_MC_step(particle)
            if (mutated_particle.tobytes() not in particles):
                repeated = False
        particles[mutated_particle.tobytes()] = 1
        
def sequentialHMC(n_particles,data):
    means = np.array([0,12.5])
    Sigma = np.matrix([[10,0],[0,20]])
    particle_list = np.random.multivariate_normal(means,Sigma,n_particles)
    particles = {}
    for particle in particle_list:
        key = particle.tobytes()
        particles[key] = 1
    global curr_data    
    for t in range(len(data)):
        curr_data = data[t] # For the target and gradient calculations.
        # Correction step.
        for key in particles:
            particle = np.frombuffer(key,dtype='float64')
            particles[key] = \
                target(particle,data[t])/\
                   kernel_density_estimate(particle,data[t],leave_out=particle)
        # Selection and mutation steps.
        selection_and_mutation(particles)
        sys.stdout.write('.');sys.stdout.flush();
        
    print("\nSequential Hamiltonian Monte Carlo: "\
        "%d%% particle acceptance rate." % (100*accepted/total))
    key_list = list(particles.keys())
    particles = [np.frombuffer(key,dtype='float64') for key in key_list]
    return particles
    
def simple_scatter(points):
    fig, axs = plt.subplots(1,figsize=(8,8))
    axs.scatter([point[0] for point in points],[point[1] for point in points],
                marker='o',s=5)
    plt.xlim((-6,6))
    plt.ylim((-3,30))

def main():
    with open('smileyface2.data', 'rb') as filehandle:
        data = pickle.load(filehandle)
    
    n_particles = 128
    groups = 4
    particles=[]
    for i in range(groups):
        particles += sequentialHMC(n_particles//groups,data)
    simple_scatter(particles)

    
main()