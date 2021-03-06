# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This is a unit test. If you would like to further develop pahmc_ode_cpu, you 
should visit here frequently. You should also be familiar with the Python (3.7)
built-in module 'unittest'.

To run this unit test, copy this file into its parent directory and run it.
"""


import cProfile, pstats

import numpy as np
import matplotlib.pyplot as plt
import time

from pahmc_ode_cpu.configure import Configure
from pahmc_ode_cpu.data_preparation import Data
from pahmc_ode_cpu.__init__ import Fetch
from pahmc_ode_cpu import lib_dynamics


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
obsdim = [1, 2, 4, 6, 8, 10, 12, 14, 15, 17, 19, 20]
# set the discretization interval
dt = 0.025

"""Specs for precision annealing and HMC."""
# set the starting Rf value
Rf0 = 1e6
# set alpha
alpha = 1.0
# set the total number of beta values
betamax = 1
# set the number of HMC samples for each beta
n_iter = 1000
# set the HMC simulation stepsize for each beta
epsilon = 1e-3
# set the number of leapfrog steps for an HMC sample for each beta
S = 100
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


"""Configure the inputs and the stimuli."""
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

stimuli = config.get_stimuli()


"""Get the dynamics object."""
try:
    Fetch.Cls = getattr(lib_dynamics, f'Builtin_{name}')
except:
    import def_dynamics
    Fetch.Cls = def_dynamics.Dynamics

from pahmc_ode_cpu.pahmc import Core

dyn = (Fetch.Cls)(name, stimuli)


"""Generate twin-experiment data."""
data_noisy, stimuli \
  = Data().generate(dyn, D, length, dt, noise, par_true, x0, burndata)
Y = data_noisy[obsdim, 0:M]


"""Truncate stimuli to fit the observation window."""
dyn.stimuli = stimuli[:, 0:M]


"""Do the calculations."""
t0 = time.perf_counter()

job = Core(dyn, Y, dt, D, obsdim, M)

cProfile.run('burn, Rm, Rf, eta_avg, acceptance, '\
             +'action, action_meanpath, ME_meanpath, FE_meanpath, '\
             +'X_init, X_gd, X_mean, par_history, par_mean, Xfinal_history '\
             +'= job.pa(Rf0, alpha, betamax, n_iter, ' \
             +'epsilon, S, mass, scaling, soft_dynrange, par_start)', \
             'profiling')

print(f'Total time = {time.perf_counter()-t0:.2f} seconds.')

p = pstats.Stats('profiling')
p.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)


"""Plot action vs. iteration."""
fig, ax = plt.subplots(figsize=(6,5))
textblue = (49/255, 99/255, 206/255)
ax.loglog(np.arange(1, n_iter+2), action[0, 1:], color=textblue, lw=1.5)
ax.set_xlim(1, n_iter+1)
ax.set_xlabel('iteration')
ax.set_ylabel('action')


"""Get an overview of performance."""
# get the action object
from pahmc_ode_cpu.utilities import Action
A = Action(dyn, Y, dt, D, obsdim, M, Rm)

# define function to extract information
def overview(X, par, Rf, scaling=1.0):
    fX = A.get_fX(X, par)

    action = A.action(X, fX, Rf)
    modelerr = np.sum(Rf/(2*A.M)*np.sum((X[:, 1:]-fX)**2, axis=1))

    gradX = A.dAdX(X, par, fX, Rf, scaling)
    gradpar = A.dAdpar(X, par, fX, Rf, scaling)

    print(f'\n       action = {action},')
    print(f'     modelerr = {modelerr},\n')
    print(f'  max |gradX| = {np.max(np.abs(gradX))},')
    print(f'  min |gradX| = {np.min(np.abs(gradX))},\n')
    print(f'max |gradpar| = {np.max(np.abs(gradpar))},')
    print(f'min |gradpar| = {np.min(np.abs(gradpar))}.')

    return action, modelerr, gradX, gradpar

# print results
print('\n--------------------------------------------------')
print('Before:')
o1_action, o1_modelerr, o1_gradX, o1_gradpar \
  = overview(X_init[0, :, :], par_history[0, 0, :], Rf[0])
print('\n--------------------------------------------------')
print('After exploration:')
o2_action, o2_modelerr, o2_gradX, o2_gradpar \
  = overview(X_gd[0, :, :], par_history[0, 1, :], Rf[0])
print('\n--------------------------------------------------')
print('After exploitation:')
o3_action, o3_modelerr, o3_gradX, o3_gradpar \
  = overview(X_mean[0, :, :], par_mean[0, :], Rf[0])

