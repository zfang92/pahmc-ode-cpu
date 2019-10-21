# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

Alternative to 'main.py', if you would like to tune the hyperparameters for  
each beta (generally a good practice), use this file as the main executable.

The user is assumed to have the following:
    1) The dynamical system. If calling one of the built-in examples, the name
    of the dynamics must have a match in 'lib_dynamics.py'; if builing from 
    scratch, 'def_dynamics.py' must be ready at this point.
    2) The data. If performing twin-experiments, the specs should be given but 
    a data file is not required; if working with real data, the data should be 
    prepared according to the user manual.
    3) If external stimuli are needed, a .npy file containing the time series; 
    4) Configuration of the code, including the hyper-parameters for PAHMC. 
    Refer to the manual for the shape and type requirements. Also note that a 
    lot of them can take either a single or an array/list of values. See user 
    manual for details.

It is suggested that the user keep a lookup table for the model paramters to 
make it easier to preserve order when working on the above steps.
"""


from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import time

from pahmc_ode_cpu.configure import Configure
from pahmc_ode_cpu.data_preparation import Data
from pahmc_ode_cpu.__init__ import Fetch
from pahmc_ode_cpu import lib_dynamics


#================type your code below (stepwise tuning)================
"""Tunable hyperparameters."""
# set the beta value to be tuned
tune_beta = 0
# set the number of HMC samples for each beta
n_iter = 500
# set the HMC simulation stepsize for each beta
epsilon = 1e-3
# set the number of leapfrog steps for an HMC sample for each beta
S = 50
# set the HMC masses for each beta
mass = (1e0, 1e0, 1e0)
# set the HMC scaling parameter for each beta
scaling = 1e5
#===================type your code below (only once)===================
"""A name for your dynamics."""
# it will be used to try to find a match in the built-ins
name = 'nakl'

"""Specs for the dynamics."""
# set the dimension of your dynamics
D = 4
# set the length of the observation window
M = 5000
# set the observed dimensions (list with smallest possible value 1)
obsdim = [1]
# set the discretization interval
dt = 0.02

"""The remaining hyperparameters."""
# set the starting Rf value
Rf0 = np.array([1.0e-1, 1.2e3, 1.6e3, 2.1e3])
# set alpha
alpha = 2.0
# set the "soft" dynamical range for initialization purpose
soft_dynrange = np.array([[-120, 0], [0, 1], [0, 1], [0, 1]])
# set an initial guess for the parameters
par_start = np.array([115, 50, 25, -70, 0.2, -55, 
                      -45, 16, 0.15, 0.4,
                      -55, -16, 1.2, 6,
                      -52, 31, 0.8, 5])

"""Specs for the twin-experiment data"""
# set the length of the data (must be greater than M defined above)
length = int(1000/dt)
# set the noise levels (standard deviations) in the data for each dimension
noise = np.array([1, 0, 0, 0])
# set the true parameters (caution: order must be consistent)
par_true = np.array([120, 50, 20, -77, 0.3, -54.4, 
                     -40, 15, 0.1, 0.4, 
                     -60, -15, 1, 7, 
                     -55, 30, 1, 5])
# set the initial condition for the data generation process
x0 = np.array([-70, 0.1, 0.9, 0.1])
# set the switch for discarding the first half of the generated data
burndata = False
#===============================end here===============================


"""Prepare current Rf and set betamax."""
Rf0 = Rf0 * (alpha ** tune_beta)
betamax = 1


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

from pahmc_ode_cpu.pahmc_tune import Core

dyn = (Fetch.Cls)(name, stimuli)


"""Generate twin-experiment data."""
data_noisy, stimuli \
  = Data().generate(dyn, D, length, dt, noise, par_true, x0, burndata)
Y = data_noisy[obsdim, 0:M]


"""Truncate stimuli to fit the observation window."""
dyn.stimuli = stimuli[:, 0:M]


"""Do the calculations."""
t0 = time.perf_counter()

job = Core(dyn, Y, dt, D, obsdim, M, tune_beta)

burn, Rm, Rf, eta_avg, acceptance, \
action, action_meanpath, ME_meanpath, FE_meanpath, \
X_init, X_gd, X_mean, par_history, par_mean, Xfinal_history \
  = job.pa(Rf0, alpha, betamax, n_iter,
           epsilon, S, mass, scaling, soft_dynrange, par_start)

print(f'Total time = {time.perf_counter()-t0:.2f} seconds.')


"""Save the results."""
np.savez(Path.cwd()/'user_results'/f'tune_{name}_{tune_beta}', 
         name=name, 
         D=D, M=M, obsdim=obsdim, dt=dt, 
         Rf0=Rf0, alpha=alpha, betamax=betamax, 
         n_iter=n_iter, epsilon=epsilon, S=S, mass=mass, scaling=scaling, 
         soft_dynrange=soft_dynrange, par_start=par_start, 
         length=length, data_noisy=data_noisy, stimuli=stimuli, 
         noise=noise, par_true=par_true, x0=x0, burndata=burndata, 
         burn=burn, 
         Rm=Rm, 
         Rf=Rf, 
         eta_avg=eta_avg, 
         acceptance=acceptance, 
         action=action, 
         action_meanpath=action_meanpath, 
         ME_meanpath=ME_meanpath, 
         FE_meanpath=FE_meanpath, 
         X_init=X_init, 
         X_gd=X_gd, 
         X_mean=X_mean, 
         par_history=par_history, 
         par_mean=par_mean, 
         Xfinal_history=Xfinal_history)


"""Plot action vs. iteration for current beta."""
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

# retrive the noiseless data (if doing twin experiment)
noiselessfile_name = Path.cwd() / 'user_data' / f'{name}_noiseless.npz'
if noiselessfile_name.exists():
    noiselessfile = np.load(noiselessfile_name)
    X_true = noiselessfile['data'][:, 0:M]
    noiselessfile.close()

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

print('\n--------------------------------------------------')
print('L1 distances (traveled and remaining):')
print('\n    from X_init to X_mean: '\
      +f'{np.sum(np.abs(X_mean[0, :, :]-X_init[0, :, :]))},')
if noiselessfile_name.exists():
    print('    from X_mean to X_true: '\
          +f'{np.sum(np.abs(X_true-X_mean[0, :, :]))},')
print('\nfrom par_init to par_mean: '\
      +f'{np.sum(np.abs(par_mean[0, :]-par_history[0, 0, :]))},')
if noiselessfile_name.exists():
    print('from par_mean to par_true: '\
          +f'{np.sum(np.abs(par_true-par_mean[0, :]))}.')

