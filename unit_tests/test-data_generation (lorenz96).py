# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This is a unit test. If you would like to further develop pahmc_ode_cpu, you 
should visit here frequently. You should also be familiar with the Python (3.7)
built-in module 'unittest'.

To run this unit test, copy this file into its parent directory and run it.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from pahmc_ode_cpu.data_preparation import Data  # import module to be tested
from pahmc_ode_cpu import lib_dynamics


name = 'lorenz96'

# specify the data below
D = 20
length = 1000
dt = 0.025
noise = 0.4 * np.ones(D)
par_true = 8.17

# stimuli = np.ones((D,2*length)) * np.arange(2*length) * 1e-9
stimuli = np.zeros((D,2*length))

x0 = np.ones(D)
x0[0] = 0.01

# instantiate
dyn = getattr(lib_dynamics, f'Builtin_{name}')(name, stimuli)

# get (noisy) data from the module being tested
t0 = time.perf_counter()
data_noisy, stimuli \
  = Data().generate(dyn, D, length, dt, noise, par_true, x0, True)
print(f'Time elapsed = {time.perf_counter()-t0:.2f} seconds.')

noiselessfile = np.load(Path.cwd()/'user_data'/f'{dyn.name}_noiseless.npz')
data_noiseless = noiselessfile['data']
noiselessfile.close()
print(f'\nChi-squared = {np.sum((data_noisy-data_noiseless)**2)}'\
      +f' ({np.sum((noise[:, np.newaxis])**2*np.ones((D,length)))} expected).')

fig, ax = plt.subplots(figsize=(8,4.5))
textred = (202/255, 51/255, 0)
textgreen = (0, 152/255, 28/255)
textblue = (49/255, 99/255, 206/255)

time = np.linspace(0, dt*length, length)
ax.plot(time, data_noisy[1, :], color=textblue, lw=1.5)

ax.legend(['data_noisy'], loc='upper right')

ax.set_xlim(0, 25)
ax.set_xticks(np.linspace(0, 25, 11))

ax.set_xlabel('Time ($\Delta t = 0.025$s)')
ax.set_ylabel('$x_1(t)$', rotation='vertical')

