# -*- coding: utf-8 -*-
"""
@author: Zheng Fang
"""


from pathlib import Path

import numpy as np


if __name__ == "__main__":
    #=========================type your code below=========================
    # name of the saved data (without the extension)
    saved  = ''
    #===============================end here===============================


def get_saved(path, saved):
    file = np.load(path/(saved+'.npz'))

    name = str(file['name'])
    D = int(file['D'])
    M = int(file['M'])
    obsdim = file['obsdim']
    dt = float(file['dt'])
    Rf0 = file['Rf0']
    alpha = float(file['alpha'])
    betamax = int(file['betamax'])
    n_iter = file['n_iter']
    epsilon = file['epsilon']
    S = file['S']
    mass = file['mass']
    scaling = file['scaling']
    soft_dynrange = file['soft_dynrange']
    par_start = file['par_start']
    length = int(file['length'])
    data_noisy = file['data_noisy']
    stimuli = file['stimuli']
    noise = file['noise']
    par_true = file['par_true']
    x0 = file['x0']
    burndata = bool(file['burndata'])
    burn = float(file['burn'])
    Rm = float(file['Rm'])
    Rf = file['Rf']
    eta_avg = file['eta_avg']
    acceptance = file['acceptance']
    action = file['action']
    action_meanpath = file['action_meanpath']
    ME_meanpath = file['ME_meanpath']
    FE_meanpath = file['FE_meanpath']
    X_init = file['X_init']
    X_gd = file['X_gd']
    X_mean = file['X_mean']
    par_history = file['par_history']
    par_mean = file['par_mean']
    Xfinal_history = file['Xfinal_history']
    
    file.close()

    return name, D, M, obsdim, dt, Rf0, alpha, betamax, \
           n_iter, epsilon, S, mass, scaling, soft_dynrange, par_start, \
           length, data_noisy, stimuli, noise, par_true, x0, burndata, \
           burn, Rm, Rf, eta_avg, acceptance, \
           action, action_meanpath, ME_meanpath, FE_meanpath, \
           X_init, X_gd, X_mean, par_history, par_mean, Xfinal_history


if __name__ == "__main__":
    name, D, M, obsdim, dt, Rf0, alpha, betamax, \
    n_iter, epsilon, S, mass, scaling, soft_dynrange, par_start, \
    length, data_noisy, stimuli, noise, par_true, x0, burndata, \
    burn, Rm, Rf, eta_avg, acceptance, \
    action, action_meanpath, ME_meanpath, FE_meanpath, \
    X_init, X_gd, X_mean, par_history, par_mean, Xfinal_history \
      = get_saved(Path.cwd(), saved)

