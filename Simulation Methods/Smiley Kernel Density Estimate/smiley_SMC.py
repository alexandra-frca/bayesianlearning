# -*- coding: utf-8 -*-
"""
Sampling from a smiley kernel density estimate using a sequential Monte Carlo 
approach. 
"""
import sys, random, pickle, matplotlib.pyplot as plt
from autograd import grad, numpy as np

dim = 2
accepted = 0
total = 0

def multivariate_gaussian(x, mu, sigma, normalize=False):
    x_mu = np.array(x-mu)
    #x_mu_t = np.transpose(x_mu) # multi_dot already transposes 
    #1-d arrays when necessary
    sigma_inv = np.linalg.inv(sigma)
    power = -0.5*np.linalg.multi_dot([x_mu,sigma_inv,x_mu])

    k = len(x)
    sigma_det = np.linalg.det(sigma)
    norm = ((2*np.pi)**k*sigma_det)**0.5
    if normalize:
        return(np.exp(power))
    else:
        return(np.exp(power)/norm)
    
def target(x,points):
    # kernel_density_estimate
    n = len(points)
    h = n**-0.2
    kde = 0
    for point in points:
        kde += multivariate_gaussian(x,point,h*np.identity(dim),
                                    normalize=True)/(n*h)
    return(kde)

def metropolis_hastings_step(point, datachunk, sigma=0.05):
    global dim, accepted, total
    Sigma = sigma*np.identity(dim)
    new_point = np.random.multivariate_normal(point, Sigma)

    p = target(new_point,datachunk)*\
        multivariate_gaussian(point,new_point,Sigma)/ \
        (target(point,datachunk)\
            *multivariate_gaussian(new_point,point,Sigma))
    a = min(1,p)

    total += 1
    if (np.random.rand() < a):
        accepted += 1
        return(new_point)
    else:
        return(point)

def selection_and_mutation(particles,datachunk):
    n = len(particles)
    selected_particles = random.choices(list(particles.keys()), 
                                      weights=particles.values(), k=n)
    particles.clear()
    for key in selected_particles:
        particle = np.frombuffer(key,dtype='float64')
        repeated = True
        while (repeated == True):
            mutated_particle = metropolis_hastings_step(particle,datachunk)
            if (mutated_particle.tobytes() not in particles):
                repeated = False
                
        particles[mutated_particle.tobytes()] = 1
        
def sequentialMC_MH(n_particles,data):
    means = np.array([0,10])
    Sigma = np.matrix([[10,0],[0,20]])
    particle_list = np.random.multivariate_normal(means,Sigma,n_particles)
    particles = {}
    for particle in particle_list:
        key = particle.tobytes()
        particles[key] = 1
    
    for t in range(len(data)):
        # Correction step.
        for key in particles:
            particle = np.frombuffer(key,dtype='float64')
            if (t==0):
                particles[key] = target(particle,data[t])\
                    /multivariate_gaussian(particle,means,Sigma,
                                    normalize=True)
            else:
                particles[key] = \
                    target(particle,data[t])/target(particle,data[t-1])    
                    
        # Selection and mutation steps.
        selection_and_mutation(particles,data[t])
        
        sys.stdout.write('.');sys.stdout.flush();
        
    print("\nSequential Monte Carlo (MH)): %d%% particle acceptance rate. " % 
          (100*accepted/total))
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
        particles += sequentialMC_MH(n_particles//groups,data)

    simple_scatter(particles)

    
main()