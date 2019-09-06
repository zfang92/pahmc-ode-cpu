# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This is a unit test. If you would like to further develop pahmc_ode_cpu, you 
should visit here frequently. You should also be familiar with the Python (3.7)
built-in module 'unittest'.

To run this unit test, copy this file into its parent directory and execute it.
"""


import time

import numpy as np

from def_dynamics import Dynamics


D = 4
M = 4000
X = np.zeros((D,M))
X[0, :] = np.random.uniform(-100.0, 50.0, M)
X[1:D, :] = np.random.uniform(0.0, 1.0, (D-1,M))
par = np.array([1.0, 120.0, 50.0, 20.0, -77.0, 0.3, -54.4, -40.0, 15, 
                0.1, 0.4, -60.0, -15, 1.0, 7.0, -55.0, 30, 1.0, 5.0])
stimuli = np.zeros((D,M))
stimuli[0, :] = np.random.uniform(0, 100, M)

dyn = Dynamics('nakl', stimuli)

vecfield = dyn.field(X, par, stimuli)
jacob = dyn.jacobian(X, par)
deriv_par = dyn.dfield_dpar(X, par)

t0 = time.perf_counter()
for i in range(10*150):
    vecfield = dyn.field(X, par, stimuli)
    jacob = dyn.jacobian(X, par)
    deriv_par = dyn.dfield_dpar(X, par)
print(f'Total time = {time.perf_counter()-t0:.2f} seconds.\n')

