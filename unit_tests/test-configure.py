# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This is a unit test. 

To run this unit test, copy this file into its parent directory and execute it.
"""


from pathlib import Path

import numpy as np

from pahmc_ode_cpu.configure import Configure


#=========================type your code below=========================
"""A name for your dynamics."""
# it will be used to try to find a match in the built-ins
name = 'lorenz96'

"""Specs for the dynamics."""
# set the dimension of your dynamics
D = 20
# set the length of the observation window
M = 200
# set the observed dimensions (list with smallest possible value 1)
obsdim = [1, 2, 4, 6, 8, 10, 12, 14, 16, 17, 19, 20]
# set the discretization interval
dt = 0.025

"""Specs for precision annealing and HMC."""
# set the starting Rf value
Rf0 = 1e6
# set alpha
alpha = 1.0
# set the total number of beta values
betamax = 3
# set the number of HMC samples for each beta
n_iter = [1, 2e3, 4]
# set the HMC simulation stepsize for each beta
epsilon = 1e-2
# set the number of leapfrog steps for an HMC sample for each beta
S = 150
# set the HMC masses for each beta
mass = (1e0, 1e0, 1e0)
# set the HMC scaling parameter for each beta
scaling = 1.0
# set the "soft" dynamical range for initialization purpose
soft_dynrange = (-10, 10)
# set an initial guess for the parameters
par_start = 8.0

"""Sepcs for the twin-experiment data"""
# set the length of the data (must be greater than M defined above)
length = 1000
# set the noise level (standard deviation) in the data
noise = 0.4
# set the true parameters (caution: order must be consistent)
par_true = 8.17
# set the initial condition for the data generation process
x0 = np.ones(D)
x0[0] = 0.01
#===============================end here===============================


"""Configuration for the inputs and the stimuli."""
config = Configure(name, 
				   D, M, obsdim, dt, 
				   Rf0, alpha, betamax, 
				   n_iter, epsilon, S, mass, scaling, 
				   soft_dynrange, par_start, 
				   length, noise, par_true, x0)

config.check_all()

name, \
D, M, obsdim, dt, \
Rf0, alpha, betamax, \
n_iter, epsilon, S, mass, scaling, \
soft_dynrange, par_start, \
length, noise, par_true, x0 = config.regulate()

file = np.load(Path.cwd()/'user_results'/'config.npz')
assert type(name) == type(str(file['name']))
assert type(D) == type(int(file['D']))
assert type(M) == type(int(file['M']))
assert type(obsdim) == type(np.array(file['obsdim'], dtype='int32'))
assert type(dt) == type(float(file['dt']))
assert type(Rf0) == type(float(file['Rf0']))
assert type(alpha) == type(float(file['alpha']))
assert type(betamax) == type(int(file['betamax']))
assert type(n_iter) == type(np.array(file['n_iter'], dtype='int32'))
assert type(epsilon) == type(np.array(file['epsilon'], dtype='float64'))
assert type(S) == type(np.array(file['S'], dtype='int32'))
assert type(mass) == type(np.array(file['mass'], dtype='float64'))
assert type(scaling) == type(np.array(file['scaling'], dtype='float64'))
assert \
  type(soft_dynrange) == type(np.array(file['soft_dynrange'], dtype='float64'))
assert type(par_start) == type(np.array(file['par_start'], dtype='float64'))
assert type(length) == type(int(file['length']))
assert type(noise) == type(float(file['noise']))
assert type(par_true) == type(np.array(file['par_true'], dtype='float64'))
assert type(x0) == type(np.array(file['x0'], dtype='float64'))

stimuli = config.get_stimuli()
