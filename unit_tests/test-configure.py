# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This is a unit test. 

To run this unit test, copy this file into its parent directory and run it.
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
n_iter = [1, 2e3, 4] * np.ones(betamax)
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

"""Specs for the twin-experiment data"""
# set the length of the data (must be greater than M defined above)
length = 1000
# set the noise levels (standard deviations) in the data for each dimension
noise = 0.4 * np.ones(D)
# set the true parameters (caution: order must be consistent)
par_true = 8.17
# set the initial condition for the data generation process
x0 = np.ones(D)
x0[0] = 0.01
# set the switch for discarding the first half of the generated data
burndata = True
#===============================end here===============================


"""Configuration for the inputs and the stimuli."""
config = Configure(name, 
                   D, M, obsdim, dt, 
                   Rf0, alpha, betamax, 
                   n_iter, epsilon, S, mass, scaling, 
                   soft_dynrange, par_start, 
                   length, noise, par_true, x0, burndata)

config.check_all()

name, \
D, M, obsdim, dt, \
Rf0, alpha, betamax, \
n_iter, epsilon, S, mass, scaling, \
soft_dynrange, par_start, \
length, noise, par_true, x0, burndata = config.regulate()

file = np.load(Path.cwd()/'user_results'/'config.npz')
assert type(name) == type(str(file['name']))
assert type(D) == type(np.int64(file['D']))
assert type(M) == type(np.int64(file['M']))
assert type(obsdim) == type(np.array(file['obsdim'], dtype='int64'))
assert type(dt) == type(float(file['dt']))
assert type(Rf0) == type(float(file['Rf0']))
assert type(alpha) == type(float(file['alpha']))
assert type(betamax) == type(np.int64(file['betamax']))
assert type(n_iter) == type(np.array(file['n_iter'], dtype='int64'))
assert type(epsilon) == type(np.array(file['epsilon'], dtype='float64'))
assert type(S) == type(np.array(file['S'], dtype='int64'))
assert type(mass) == type(np.array(file['mass'], dtype='float64'))
assert type(scaling) == type(np.array(file['scaling'], dtype='float64'))
assert \
  type(soft_dynrange) == type(np.array(file['soft_dynrange'], dtype='float64'))
assert type(par_start) == type(np.array(file['par_start'], dtype='float64'))
assert type(length) == type(np.int64(file['length']))
assert type(noise) == type(np.array(file['noise'], dtype='float64'))
assert type(par_true) == type(np.array(file['par_true'], dtype='float64'))
assert type(x0) == type(np.array(file['x0'], dtype='float64'))
assert type(burndata) == type(bool(file['burndata']))

stimuli = config.get_stimuli()

print('\nTest finished.')

