# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This is a unit test. If you would like to further develop pahmc_ode_cpu, you 
should visit here frequently. You should also be familiar with the Python (3.7)
built-in module 'unittest'.

To run this unit test, copy this file into its parent directory and run it.
"""


import numpy as np
import torch as th

from pahmc_ode_cpu import lib_dynamics


#=========================type your code below=========================
name = 'nakl'

D = 4
M = 10

X = np.zeros((D,M))
X[0, :] = np.random.uniform(-100.0, 50.0, M)
X[1:D, :] = np.random.uniform(0.0, 1.0, (D-1,M))

par = np.array([120.0, 50.0, 20.0, -77.0, 0.3, -54.4, -40.0, 15, 
                0.1, 0.4, -60.0, -15, 1.0, 7.0, -55.0, 30, 1.0, 5.0])

stimulus = np.zeros((D,M))
stimulus[0, :] = np.random.uniform(-30, 30, M)
#===============================end here===============================

X = th.from_numpy(X)
X.requires_grad = True

par = th.from_numpy(par)
par.requires_grad = True

stimulus = th.from_numpy(stimulus)

vecfield = th.zeros(D,M)

#=========================type your code below=========================
vecfield[0, :] \
  = stimulus[0, :] \
    + par[0] * (X[1, :] ** 3) * X[2, :] * (par[1] - X[0, :]) \
    + par[2] * (X[3, :] ** 4) * (par[3] - X[0, :]) \
    + par[4] * (par[5] - X[0, :])

tanh_m = th.tanh((X[0, :]-par[6])/par[7])
eta_m = 1 / 2 * (1 + tanh_m)
tau_m = par[8] + par[9] * (1 - tanh_m * tanh_m)
vecfield[1, :] = (eta_m - X[1, :]) / tau_m

tanh_h = th.tanh((X[0, :]-par[10])/par[11])
eta_h = 1 / 2 * (1 + tanh_h)
tau_h = par[12] + par[13] * (1 - tanh_h * tanh_h)
vecfield[2, :] = (eta_h - X[2, :]) / tau_h

tanh_n = th.tanh((X[0, :]-par[14])/par[15])
eta_n = 1 / 2 * (1 + tanh_n)
tau_n = par[16] + par[17] * (1 - tanh_n * tanh_n)
vecfield[3, :] = (eta_n - X[3, :]) / tau_n
#===============================end here===============================

scalarfield = th.sum(vecfield)
scalarfield.backward()

compare_jacob_auto = np.around(X.grad.numpy(), decimals=6)
compare_derivpar_auto = np.around(par.grad.numpy(), decimals=6)


print('\nTesting... ', end='')
X = X.detach().numpy()
par = par.detach().numpy()
stimulus = stimulus.numpy()

try:
    dyn = getattr(lib_dynamics, f'Builtin_{name}')(name, stimulus)
except:
    import def_dynamics
    dyn = def_dynamics.Dynamics(name, stimulus)

compare_jacob_manual = np.sum(dyn.jacobian(X, par), axis=0)
compare_jacob_manual = np.around(compare_jacob_manual, decimals=6)
compare_derivpar_manual = np.sum(dyn.dfield_dpar(X, par), axis=(0,1))
compare_derivpar_manual = np.around(compare_derivpar_manual, decimals=6)

assert np.array_equal(compare_jacob_auto, compare_jacob_manual), \
    "Something's wrong with the Jacobian."
assert np.array_equal(compare_derivpar_auto, compare_derivpar_manual), \
    "Something's wrong with the derivatives w.r.t. the parameters."

print('ok.')

