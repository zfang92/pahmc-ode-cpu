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

from pahmc_ode_cpu.data_preparation import Data
from def_dynamics import Dynamics  # change this later


name = 'nakl'

# specs
D = 4
dt = 0.02
length = int(1000/dt)
noise = np.array([1, 0, 0, 0], dtype='float64')
par_true = np.array([1, 
                     120, 50, 20, -77, 0.3, -54.4, 
                     -40, 15, 0.1, 0.4, 
                     -60, -15, 1, 7, 
                     -55, 30, 1, 5], dtype='float64')
x0 = np.array([-70, 0.1, 0.9, 0.1], dtype='float64')
burndata = False

stimuli = np.load(Path.cwd()/'user_data'/f'{name}_stimuli.npy')[:, 0:2*length]


# generate the data
dyn = Dynamics(name, stimuli)

t0 = time.perf_counter()
data_noisy, stimuli \
  = Data().generate(dyn, D, length, dt, noise, par_true, x0, burndata)
print(f'Time elapsed = {time.perf_counter()-t0:.2f} seconds.')


# noise level
noiselessfile = np.load(Path.cwd()/'user_data'/f'{dyn.name}_noiseless.npz')
data_noiseless = noiselessfile['data']
noiselessfile.close()
print(f'\nChi-squared = {np.sum((data_noisy-data_noiseless)**2)}'\
      +f' ({np.sum((noise[:, np.newaxis])**2*np.ones((D,length)))} expected).')

# plot
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12,8))
textred = (202/255, 51/255, 0)
textblue = (49/255, 99/255, 206/255)

time = np.linspace(0, int(length*dt)-dt, length)

ax1.plot(time, data_noisy[0, :], color=textblue)
ax1.set_xlim(0, int(length*dt)-dt)
ax1.set_xticks(np.linspace(0, 1000, 11))
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('V(t)', rotation='horizontal')

ax2.plot(time, stimuli[0, :], color=textred)
ax2.set_xlim(0, int(length*dt)-dt)
ax2.set_xticks(np.linspace(0, 1000, 11))
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('I(t)', rotation='horizontal')

