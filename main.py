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
from pahmc_ode_cpu.data_preparation import Data
from pahmc_ode_cpu.__init__ import Fetch
from pahmc_ode_cpu import lib_dynamics


#=========================type your code below=========================
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

"""Specs for precision annealing and HMC."""
# set the starting Rf value
Rf0 = np.array([1.0e-1, 1.2e3, 1.6e3, 2.1e3])
# set alpha
alpha = 2.0
# set the total number of beta values
betamax = 20
# set the number of HMC samples for each beta
n_iter = np.concatenate((500*np.ones(19), 
                         1000*np.ones(1)))
# set the HMC simulation stepsize for each beta
epsilon = np.concatenate((1e-3*np.ones(1), 
                          1e-4*np.ones(6), 
                          1e-5*np.ones(3), 
                          1e-6*np.ones(2),
                          1e-7*np.ones(4),
                          1e-8*np.ones(3),
                          1e-9*np.ones(1)))
# set the number of leapfrog steps for an HMC sample for each beta
S = np.concatenate((50*np.ones(1), 
                    100*np.ones(9),
                    200*np.ones(2),
                    300*np.ones(2),
                    500*np.ones(2),
                    700*np.ones(1),
                    800*np.ones(1),
                    1000*np.ones(1),
                    1500*np.ones(1)))
# set the HMC masses for each beta
mass = (1e0,1e0,1e0)
# set the HMC scaling parameter for each beta
scaling = np.concatenate((1e5*np.ones(1),
                          1e6*np.ones(6),
                          5e7*np.ones(1),
                          3e7*np.ones(1),
                          2e7*np.ones(1),
                          1e9*np.ones(1),
                          5e8*np.ones(1),
                          2e10*np.ones(1),
                          1e10*np.ones(1),
                          6e9*np.ones(1),
                          3e9*np.ones(1),
                          1e11*np.ones(1),
                          5e10*np.ones(1),
                          3e10*np.ones(1),
                          1.5e12*np.ones(1)))
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

burn, Rm, Rf, eta_avg, acceptance, \
action, action_meanpath, ME_meanpath, FE_meanpath, \
X_init, X_gd, X_mean, par_history, par_mean, Xfinal_history \
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

