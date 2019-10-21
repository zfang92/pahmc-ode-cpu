# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This is a unit test. 

To run this unit test, copy this file into its parent directory and run it.
"""


from pathlib import Path
import time

import numpy as np

from pahmc_ode_cpu.__init__ import Fetch
from pahmc_ode_cpu import lib_dynamics
from user_results import read


"""Get necessary variables."""
name, D, M, obsdim, dt, Rf0, alpha, betamax, \
n_iter, epsilon, S, mass, scaling, soft_dynrange, par_start, \
length, data_noisy, stimuli, noise, par_true, x0, burndata, \
burn, Rm, Rf, eta_avg, acceptance, \
action, action_meanpath, ME_meanpath, FE_meanpath, \
X_init, X_gd, X_mean, par_history, par_mean, Xfinal_history \
  = read.get_saved(Path.cwd()/'unit_tests', 'test-gd')

unobsdim = np.int64(np.setdiff1d(np.arange(D), obsdim))
mass_X = np.zeros((betamax,D,M))
mass_par = np.zeros((betamax,len(par_start)))
for beta in range(betamax):
    mass_X[beta, obsdim, :] = mass[beta, 0]
    mass_X[beta, unobsdim, :] = mass[beta, 1]
    mass_par[beta, :] = mass[beta, 2]


"""Some preparation work."""
try:
    Fetch.Cls = getattr(lib_dynamics, f'Builtin_{name}')
except:
    import def_dynamics
    Fetch.Cls = def_dynamics.Dynamics

from pahmc_ode_cpu.utilities import Action
from pahmc_ode_cpu.pahmc import MC

dyn = (Fetch.Cls)(name, stimuli[:, 0:M])
Y = data_noisy[obsdim, 0:M]

A = Action(dyn, Y, dt, D, obsdim, M, Rm)
mc = MC(D, obsdim, unobsdim, M, A, Rf, \
        epsilon, S, mass, scaling, mass_X, mass_par)


"""Get an overview."""
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


"""Show results."""
print('\n--------------------------------------------------')
print('Initial values before gradient descent:')
ov_action, ov_modelerr, ov_gradX, ov_gradpar \
  = overview(X_init[0, :, :], par_history[0, 0, :], Rf[0])

t0 = time.perf_counter()
X_gd, par_gd, action_gd, eta \
  = mc.gd(X_init[0, :, :], par_history[0, 0, :], Rf[0], 0.1, 1000)
print(f'Total time = {time.perf_counter()-t0:.2f} seconds.')

print('\n--------------------------------------------------')
print('After gradient descent:')
ow_action, ow_modelerr, ow_gradX, ow_gradpar \
  = overview(X_gd, par_gd, Rf[0])

