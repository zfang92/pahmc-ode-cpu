# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This is the main executable of pahmc_ode_cpu and should be the point of entry
at which all the necessary information is provided. In particular, the user 
is assumed to have the following:
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


from datetime import date
from pathlib import Path

import numpy as np
import time

from pahmc_ode_cpu.configure import Configure
from pahmc_ode_cpu.pahmc import Core
from pahmc_ode_cpu.data_preparation import Data
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
obsdim = [1, 4, 7, 9, 11, 13, 16, 19]
# set the discretization interval
dt = 0.025

"""Specs for precision annealing and HMC."""
# set the starting Rf value
Rf0 = 0.1
# set alpha
alpha = 1.8
# set the total number of beta values
betamax = 25
# set the number of HMC samples for each beta
n_iter = [1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3,  
          1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 
          1e3, 1e3, 1e4, 1e4, 1e4]
# set the HMC simulation stepsize for each beta
epsilon = [1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 
           1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 
           1e-4, 1e-4, 1e-5, 1e-5, 1e-5]
# set the number of leapfrog steps for an HMC sample for each beta
S = [150, 150, 150, 150, 150, 50, 50, 50, 50, 50, 
     50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 
     50, 50, 50, 50, 50]
# set the HMC masses for each beta
mass = [(1e0, 1e0, 1e0), (1e0, 1e0, 1e0), (1e0, 1e0, 1e0), (1e0, 1e0, 1e0), 
        (1e0, 1e0, 1e0), (1e0, 1e0, 1e0), (1e0, 1e0, 1e0), (1e0, 1e0, 1e0), 
        (1e0, 1e0, 1e0), (1e0, 1e0, 1e0), (1e0, 1e0, 1e0), (1e0, 1e0, 1e0), 
        (1e0, 1e0, 1e0), (1e1, 1e1, 1e1), (1e1, 1e1, 1e1), (1e1, 1e1, 1e1), 
        (1e1, 1e1, 1e1), (1e1, 1e1, 1e1), (1e2, 1e2, 1e2), (1e2, 1e2, 1e2), 
        (1e2, 1e2, 1e2), (1e2, 1e2, 1e2), (1e1, 1e1, 1e1), (1e1, 1e1, 1e1), 
        (1e1, 1e1, 1e1)]
# set the HMC scaling parameter for each beta
scaling = 1e6
# set the "soft" dynamical range for initialization purpose
soft_dynrange = (-10, 10)
# set an initial guess for the parameters
par_start = 8.0

"""Sepcs for the twin-experiment data"""
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
    dyn = getattr(lib_dynamics, f'Builtin_{name}')(name, stimuli)
except:
    import def_dynamics
    dyn = def_dynamics.Dynamics(name, stimuli)


"""Generate twin-experiment data."""
data_noisy, stimuli \
  = Data().generate(dyn, D, length, dt, noise, par_true, x0, burndata)
Y = data_noisy[obsdim, 0:M]


"""Truncate stimuli to fit the observation window."""
dyn.stimuli = stimuli[:, 0:M]


"""Do the calculations."""
t0 = time.perf_counter()

job = Core(dyn, Y, dt, D, obsdim, M)

acceptance, action, action_meanpath, burn, \
FE_meanpath, ME_meanpath, par_history, par_mean, \
Rf, Rm, X_init, X_mean, Xfinal_history \
  = job.pa(Rf0, alpha, betamax, n_iter,
           epsilon, S, mass, scaling, soft_dynrange, par_start)

print(f'Total time = {time.perf_counter()-t0:.2f} seconds.')


"""Save the results."""
day = date.today().strftime('%Y-%m-%d')
i = 1
while (Path.cwd() / 'user_results' / f'{name}_{day}_{i}.npz').exists():
    i = i + 1

np.savez(Path.cwd()/'user_results'/f'{name}_{day}_{i}', 
         name=name, 
         D=D, M=M, obsdim=obsdim, dt=dt, 
         Rf0=Rf0, alpha=alpha, betamax=betamax, 
         n_iter=n_iter, epsilon=epsilon, S=S, mass=mass, scaling=scaling, 
         soft_dynrange=soft_dynrange, par_start=par_start, 
         length=length, 
         noise=noise, par_true=par_true, x0=x0, burndata=burndata, 
         acceptance=acceptance, 
         action=action, 
         action_meanpath=action_meanpath, 
         burn=burn, 
         FE_meanpath=FE_meanpath, 
         ME_meanpath=ME_meanpath, 
         par_history=par_history, 
         par_mean=par_mean, 
         Rf=Rf, 
         Rm=Rm, 
         X_init=X_init, 
         X_mean=X_mean, 
         Xfinal_history=Xfinal_history)

