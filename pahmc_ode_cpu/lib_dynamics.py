# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This module contains all the built-in dynamics, each being a class, that are 
ready for deployment. If the user has input a name (should be all lowercase) 
that has a match here, the 'def_dynamics' module will be ignored and the 
corresponding class here will be instantiated.

The name of each class below begins with 'Builtin_' and appends the name of 
the dynamics after that. Future added classes should be named this way.

Three, and only three methods need to be implemented for each class. See below.
    1) field(self, X, par, stimulus):
        Inputs
        ------
               X: D-by-M numpy array for any positive integer M.
             par: one-dimensional (shapeless) numpy array.
        stimulus: D-by-M numpy array for any positive integer M; stimulus is 
                  a subset of 'self.stimuli'.

        Returns
        -------
        D-by-M numpy array for any positive integer M. This should include the 
        external stimuli, if any.

    2) jacobian(self, X, par):
        Inputs
        ------
          X: D-by-M numpy array for any positive integer M.
        par: one-dimensional (shapeless) numpy array.

        Returns
        -------
        D-by-D-by-M numpy array for any positive integer M.

    3) dfield_dpar(self, X, par):
        Inputs
        ------
          X: D-by-M numpy array for any positive integer M.
        par: one-dimensional (shapeless) numpy array.

        Returns
        -------
        D-by-M-by-len(par) numpy array. Each index in the third axis 
        corresponds to a D-by-M numpy array that contains the derivatives with 
        respect to the path X.
"""


from numba import jitclass, types
import numpy as np


spec = [('name', types.string), ('stimuli', types.float64[:, :])]


@jitclass(spec)
class Builtin_lorenz96:
    """
    This class implements the standard Lorenz96 model. Fortunately, there is
    only one representation of the model.
    """

    def __init__(self, name, stimuli):
        self.name = name
        self.stimuli = stimuli

    def field(self, X, par, stimulus):
        (D, M) = np.shape(X)
        vecfield = np.zeros((D,M))

        for m in range(M):
            vecfield[0, m] = (X[1, m] - X[D-2, m]) * X[D-1, m] - X[0, m]
            vecfield[1, m] = (X[2, m] - X[D-1, m]) * X[0, m] - X[1, m]
            vecfield[D-1, m] = (X[0, m] - X[D-3, m]) * X[D-2, m] - X[D-1, m]
            for a in range(2, D-1):
                vecfield[a, m] = (X[a+1, m] - X[a-2, m]) * X[a-1, m] - X[a, m]
        
        return vecfield + par[0]

    def jacobian(self, X, par):
        (D, M) = np.shape(X)
        jacob = np.zeros((D,D,M))

        for m in range(M):
            for i in range(1, D+1):
                for j in range(1, D+1):
                    jacob[i-1, j-1, m] \
                      = (1 + (i - 2) % D == j) \
                        * (X[i%D, m] - X[(i-3)%D, m]) \
                        + ((1 + i % D == j) - (1 + (i - 3) % D == j)) \
                        * X[(i-2)%D, m] - (i == j)
        
        return jacob

    def dfield_dpar(self, X, par):
        (D, M) = np.shape(X)

        return np.ones((D,M,len(par)))


@jitclass(spec)
class Builtin_nakl:
    """
    This class implements the Hodgkin-Huxley model as described in Toth et al., 
    Biological Cybernetics (2011). It has 19 parameters as follows:
    g_Na, E_Na, g_K, E_K, g_L, E_L; 
    Vm, dVm, tau_m0, tau_m1; 
    Vh, dVh, tau_h0, tau_h1;
    Vn, dVn, tau_n0, tau_n1.
    """

    def __init__(self, name, stimuli):
        self.name = name
        self.stimuli = stimuli

    def field(self, X, par, stimulus):
        (D, M) = np.shape(X)
        vecfield = np.zeros((D,M))

        vecfield[0, :] \
          = stimulus[0, :] \
            + par[0] * (X[1, :] ** 3) * X[2, :] * (par[1] - X[0, :]) \
            + par[2] * (X[3, :] ** 4) * (par[3] - X[0, :]) \
            + par[4] * (par[5] - X[0, :])

        tanh_m = np.tanh((X[0, :]-par[6])/par[7])
        eta_m = 1 / 2 * (1 + tanh_m)
        tau_m = par[8] + par[9] * (1 - tanh_m * tanh_m)
        vecfield[1, :] = (eta_m - X[1, :]) / tau_m

        tanh_h = np.tanh((X[0, :]-par[10])/par[11])
        eta_h = 1 / 2 * (1 + tanh_h)
        tau_h = par[12] + par[13] * (1 - tanh_h * tanh_h)
        vecfield[2, :] = (eta_h - X[2, :]) / tau_h

        tanh_n = np.tanh((X[0, :]-par[14])/par[15])
        eta_n = 1 / 2 * (1 + tanh_n)
        tau_n = par[16] + par[17] * (1 - tanh_n * tanh_n)
        vecfield[3, :] = (eta_n - X[3, :]) / tau_n
        
        return vecfield

    def jacobian(self, X, par):
        (D, M) = np.shape(X)
        jacob = np.zeros((D,D,M))

        jacob[0, 0, :] = - par[0] * (X[1, :] ** 3) * X[2, :] \
                         - par[2] * (X[3, :] ** 4) - par[4]

        jacob[0, 1, :] \
          = 3 * par[0] * (X[1, :] ** 2) * X[2, :] * (par[1] - X[0, :])

        jacob[0, 2, :] = par[0] * (X[1, :] ** 3) * (par[1] - X[0, :])

        jacob[0, 3, :] = 4 * par[2] * (X[3, :] ** 3) * (par[3] - X[0, :])

        tanh_m = np.tanh((X[0, :]-par[6])/par[7])
        kernel_m = (1 - tanh_m * tanh_m)
        eta_m = 1 / 2 * (1 + tanh_m)
        tau_m = par[8] + par[9] * kernel_m
        eta_der_m = 1 / (2 * par[7]) * kernel_m
        tau_der_m = - 2 * par[9] / par[7] * tanh_m * kernel_m
        jacob[1, 0, :] \
          = eta_der_m / tau_m + tau_der_m * (X[1, :] - eta_m) / (tau_m * tau_m)

        tanh_h = np.tanh((X[0, :]-par[10])/par[11])
        kernel_h = (1 - tanh_h * tanh_h)
        eta_h = 1 / 2 * (1 + tanh_h)
        tau_h = par[12] + par[13] * kernel_h
        eta_der_h = 1 / (2 * par[11]) * kernel_h
        tau_der_h = - 2 * par[13] / par[11] * tanh_h * kernel_h
        jacob[2, 0, :] \
          = eta_der_h / tau_h + tau_der_h * (X[2, :] - eta_h) / (tau_h * tau_h)

        tanh_n = np.tanh((X[0, :]-par[14])/par[15])
        kernel_n = (1 - tanh_n * tanh_n)
        eta_n = 1 / 2 * (1 + tanh_n)
        tau_n = par[16] + par[17] * kernel_n
        eta_der_n = 1 / (2 * par[15]) * kernel_n
        tau_der_n = - 2 * par[17] / par[15] * tanh_n * kernel_n
        jacob[3, 0, :] \
          = eta_der_n / tau_n + tau_der_n * (X[3, :] - eta_n) / (tau_n * tau_n)

        jacob[1, 1, :] = - 1 / tau_m

        jacob[2, 2, :] = - 1 / tau_h

        jacob[3, 3, :] = - 1 / tau_n
        
        return jacob

    def dfield_dpar(self, X, par):
        (D, M) = np.shape(X)
        deriv_par = np.zeros((D,M,len(par)))

        deriv_par[0, :, 0] = (X[1, :] ** 3) * X[2, :] * (par[1] - X[0, :])

        deriv_par[0, :, 1] = par[0] * (X[1, :] ** 3) * X[2, :]

        deriv_par[0, :, 2] = (X[3, :] ** 4) * (par[3] - X[0, :])

        deriv_par[0, :, 3] = par[2] * (X[3, :] ** 4)

        deriv_par[0, :, 4] = par[5] - X[0, :]

        deriv_par[0, :, 5] = par[4]

        tanh_m = np.tanh((X[0, :]-par[6])/par[7])
        kernel_m = (1 - tanh_m * tanh_m)
        eta_m = 1 / 2 * (1 + tanh_m)
        tau_m = par[8] + par[9] * kernel_m
        common_m = (X[1, :] - eta_m) / (tau_m * tau_m)
        eta_der_m = - 1 / (2 * par[7]) * kernel_m
        tau_der_m = 2 * par[9] / par[7] * tanh_m * kernel_m
        deriv_par[1, :, 6] = eta_der_m / tau_m + tau_der_m * common_m

        eta_der_m = - (X[0, :] - par[6]) / (2 * (par[7] ** 2)) * kernel_m
        tau_der_m = 2 * par[9] * (X[0, :] - par[6]) / (par[7] ** 2) \
                    * tanh_m * kernel_m
        deriv_par[1, :, 7] = eta_der_m / tau_m + tau_der_m * common_m

        deriv_par[1, :, 8] = common_m

        deriv_par[1, :, 9] = kernel_m * common_m

        tanh_h = np.tanh((X[0, :]-par[10])/par[11])
        kernel_h = (1 - tanh_h * tanh_h)
        eta_h = 1 / 2 * (1 + tanh_h)
        tau_h = par[12] + par[13] * kernel_h
        common_h = (X[2, :] - eta_h) / (tau_h * tau_h)
        eta_der_h = - 1 / (2 * par[11]) * kernel_h
        tau_der_h = 2 * par[13] / par[11] * tanh_h * kernel_h
        deriv_par[2, :, 10] = eta_der_h / tau_h + tau_der_h * common_h

        eta_der_h = - (X[0, :] - par[10]) / (2 * (par[11] ** 2)) * kernel_h
        tau_der_h = 2 * par[13] * (X[0, :] - par[10]) / (par[11] ** 2) \
                    * tanh_h * kernel_h
        deriv_par[2, :, 11] = eta_der_h / tau_h + tau_der_h * common_h

        deriv_par[2, :, 12] = common_h

        deriv_par[2, :, 13] = kernel_h * common_h

        tanh_n = np.tanh((X[0, :]-par[14])/par[15])
        kernel_n = (1 - tanh_n * tanh_n)
        eta_n = 1 / 2 * (1 + tanh_n)
        tau_n = par[16] + par[17] * kernel_n
        common_n = (X[3, :] - eta_n) / (tau_n * tau_n)
        eta_der_n = - 1 / (2 * par[15]) * kernel_n
        tau_der_n = 2 * par[17] / par[15] * tanh_n * kernel_n
        deriv_par[3, :, 14] = eta_der_n / tau_n + tau_der_n * common_n

        eta_der_n = - (X[0, :] - par[14]) / (2 * (par[15] ** 2)) * kernel_n
        tau_der_n = 2 * par[17] * (X[0, :] - par[14]) / (par[15] ** 2) \
                    * tanh_n * kernel_n
        deriv_par[3, :, 15] = eta_der_n / tau_n + tau_der_n * common_n

        deriv_par[3, :, 16] = common_n

        deriv_par[3, :, 17] = kernel_n * common_n
        
        return deriv_par

