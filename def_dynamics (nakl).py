# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

Whenever you have a new dynamical system which you would like to run with
pahmc_ode_cpu, this file should be your first stop. However, if your new
system is already included in the built-in examples (see user manual), you do
not need to touch this file.

Note that this is neither an executable nor the main program. Instead, this is
where you should tell pahmc_ode_cpu about the details of your dynamical system.

In this file, between each pair of lines that look like
'#=========================type your code below=========================
 #===============================end here===============================',
write down,
    1) The vector field of your dynamical system including external stimuli;
    2) Its Jacobian;
    3) The derivatives with respect to each parameter of the system.

Double check, and pay special attention to the shape requirements for the 
intputs and outputs. Mistakes made here are not easily detectable by looking at 
the outputs. When in doubt, check with the user manual. You can use the 
corresponding file in 'unit_tests' or write your own unit test module if you 
prefer debugging this way.
"""


from numba import jitclass, types
import numpy as np


spec = [('name', types.string), ('stimuli', types.float64[:, :])]
@jitclass(spec)
class Dynamics:
    """
    This should contain all that pahmc_ode_cpu needs to know about your 
    dynamical system.
    """

    def __init__(self, name, stimuli):
        """
        This class defines the vector field.

        Inputs
        ------
           name: string specifying the name of the dynamics.
        stimuli: 2D array of external stimuli.
        """
        self.name = name
        self.stimuli = stimuli

    def field(self, X, par, stimulus):
        """
        This is your vector field.

        Parameters live in 'par'. You will later initialize 'par' in the main 
        script. You may choose an arbitrary order inside 'par', but you should 
        keep it consistent throughout.

        Inputs
        ------
               X: D-by-M numpy array for any positive integer M.
             par: one-dimensional (shapeless) numpy array.
        stimulus: D-by-M numpy array for any positive integer M; stimulus is 
                  a subset of 'self.stimuli'.

        Returns
        -------
        vecfield: D-by-M numpy array for any positive integer M. 
                  Caution: make sure to include external stimulus, if any.
        """
        (D, M) = np.shape(X)
        vecfield = np.zeros((D,M))  # initialize the output (with stimulus)

        #=========================type your code below=========================
        vecfield[0, :] \
          = stimulus[0, :] / par[0] \
            + par[1] / par[0] * (X[1, :] ** 3) * X[2, :] * (par[2] - X[0, :]) \
            + par[3] / par[0] * (X[3, :] ** 4) * (par[4] - X[0, :]) \
            + par[5] / par[0] * (par[6] - X[0, :])

        tanh_m = np.tanh((X[0, :]-par[7])/par[8])
        eta_m = 1 / 2 * (1 + tanh_m)
        tau_m = par[9] + par[10] * (1 - tanh_m * tanh_m)
        vecfield[1, :] = (eta_m - X[1, :]) / tau_m

        tanh_h = np.tanh((X[0, :]-par[11])/par[12])
        eta_h = 1 / 2 * (1 + tanh_h)
        tau_h = par[13] + par[14] * (1 - tanh_h * tanh_h)
        vecfield[2, :] = (eta_h - X[2, :]) / tau_h

        tanh_n = np.tanh((X[0, :]-par[15])/par[16])
        eta_n = 1 / 2 * (1 + tanh_n)
        tau_n = par[17] + par[18] * (1 - tanh_n * tanh_n)
        vecfield[3, :] = (eta_n - X[3, :]) / tau_n
        #===============================end here===============================
        return vecfield

    def jacobian(self, X, par):
        """
        This is the Jacobian of your vector field.

        Inputs
        ------
          X: D-by-M numpy array for any positive integer M.
        par: one-dimensional (shapeless) numpy array.

        Returns
        -------
        jacob: D-by-D-by-M numpy array for any positive integer M.
        """
        (D, M) = np.shape(X)
        idenmat = np.identity(D)
        jacob = np.zeros((D,D,M))  # initialize the output

        #=========================type your code below=========================
        jacob[0, 0, :] \
          = - par[1] / par[0] * (X[1, :] ** 3) * X[2, :] \
            - par[3] / par[0] * (X[3, :] ** 4) - par[5] / par[0]

        jacob[0, 1, :] \
          = 3 * par[1] / par[0] * (X[1, :] ** 2) * X[2, :] * (par[2] - X[0, :])

        jacob[0, 2, :] = par[1] / par[0] * (X[1, :] ** 3) * (par[2] - X[0, :])

        jacob[0, 3, :] \
          = 4 * par[3] / par[0] * (X[3, :] ** 3) * (par[4] - X[0, :])

        tanh_m = np.tanh((X[0, :]-par[7])/par[8])
        kernel_m = (1 - tanh_m * tanh_m)
        eta_m = 1 / 2 * (1 + tanh_m)
        tau_m = par[9] + par[10] * kernel_m
        eta_der_m = 1 / (2 * par[8]) * kernel_m
        tau_der_m = - 2 * par[10] / par[8] * tanh_m * kernel_m
        jacob[1, 0, :] \
          = eta_der_m / tau_m + tau_der_m * (X[1, :] - eta_m) / (tau_m * tau_m)

        tanh_h = np.tanh((X[0, :]-par[11])/par[12])
        kernel_h = (1 - tanh_h * tanh_h)
        eta_h = 1 / 2 * (1 + tanh_h)
        tau_h = par[13] + par[14] * kernel_h
        eta_der_h = 1 / (2 * par[12]) * kernel_h
        tau_der_h = - 2 * par[14] / par[12] * tanh_h * kernel_h
        jacob[2, 0, :] \
          = eta_der_h / tau_h + tau_der_h * (X[2, :] - eta_h) / (tau_h * tau_h)

        tanh_n = np.tanh((X[0, :]-par[15])/par[16])
        kernel_n = (1 - tanh_n * tanh_n)
        eta_n = 1 / 2 * (1 + tanh_n)
        tau_n = par[17] + par[18] * kernel_n
        eta_der_n = 1 / (2 * par[16]) * kernel_n
        tau_der_n = - 2 * par[18] / par[16] * tanh_n * kernel_n
        jacob[3, 0, :] \
          = eta_der_n / tau_n + tau_der_n * (X[3, :] - eta_n) / (tau_n * tau_n)

        jacob[1, 1, :] = - 1 / tau_m

        jacob[2, 2, :] = - 1 / tau_h

        jacob[3, 3, :] = - 1 / tau_n
        #===============================end here===============================
        return jacob

    def dfield_dpar(self, X, par):
        """
        This contains the derivatives of your vector field on the parameters.
        When constructing 'deriv_par', you should use the same order for the 
        output, 'deriv_par', as in 'par'.

        Inputs
        ------
          X: D-by-M numpy array for any positive integer M.
        par: one-dimensional (shapeless) numpy array.

        Returns
        -------
        deriv_par: D-by-M-by-len(par) numpy array. Each index in the third axis
                   corresponds to a D-by-M numpy array that contains the 
                   derivatives with respect to the path X.
        """
        (D, M) = np.shape(X)
        deriv_par = np.zeros((D,M,len(par)))  # initialize the output

        #=========================type your code below=========================
        deriv_par[0, :, 0] \
          = - 1 / (par[0] ** 2) \
                * (self.stimuli[0, :] \
                   + par[1] * (X[1, :] ** 3) * X[2, :] * (par[2] - X[0, :]) \
                   + par[3] * (X[3, :] ** 4) * (par[4] - X[0, :]) \
                   + par[5] * (par[6] - X[0, :]))

        deriv_par[0, :, 1] \
          = 1 / par[0] * (X[1, :] ** 3) * X[2, :] * (par[2] - X[0, :])

        deriv_par[0, :, 2] = par[1] / par[0] * (X[1, :] ** 3) * X[2, :]

        deriv_par[0, :, 3] = 1 / par[0] * (X[3, :] ** 4) * (par[4] - X[0, :])

        deriv_par[0, :, 4] = par[3] / par[0] * (X[3, :] ** 4)

        deriv_par[0, :, 5] = 1 / par[0] * (par[6] - X[0, :])

        deriv_par[0, :, 6] = par[5] / par[0]

        tanh_m = np.tanh((X[0, :]-par[7])/par[8])
        kernel_m = (1 - tanh_m * tanh_m)
        eta_m = 1 / 2 * (1 + tanh_m)
        tau_m = par[9] + par[10] * kernel_m
        common_m = (X[1, :] - eta_m) / (tau_m * tau_m)
        eta_der_m = - 1 / (2 * par[8]) * kernel_m
        tau_der_m = 2 * par[10] / par[8] * tanh_m * kernel_m
        deriv_par[1, :, 7] = eta_der_m / tau_m + tau_der_m * common_m

        eta_der_m = - (X[0, :] - par[7]) / (2 * (par[8] ** 2)) * kernel_m
        tau_der_m = 2 * par[10] * (X[0, :] - par[7]) / (par[8] ** 2) \
                    * tanh_m * kernel_m
        deriv_par[1, :, 8] = eta_der_m / tau_m + tau_der_m * common_m

        deriv_par[1, :, 9] = common_m

        deriv_par[1, :, 10] = kernel_m * common_m

        tanh_h = np.tanh((X[0, :]-par[11])/par[12])
        kernel_h = (1 - tanh_h * tanh_h)
        eta_h = 1 / 2 * (1 + tanh_h)
        tau_h = par[13] + par[14] * kernel_h
        common_h = (X[2, :] - eta_h) / (tau_h * tau_h)
        eta_der_h = - 1 / (2 * par[12]) * kernel_h
        tau_der_h = 2 * par[14] / par[12] * tanh_h * kernel_h
        deriv_par[2, :, 11] = eta_der_h / tau_h + tau_der_h * common_h

        eta_der_h = - (X[0, :] - par[11]) / (2 * (par[12] ** 2)) * kernel_h
        tau_der_h = 2 * par[14] * (X[0, :] - par[11]) / (par[12] ** 2) \
                    * tanh_h * kernel_h
        deriv_par[2, :, 12] = eta_der_h / tau_h + tau_der_h * common_h

        deriv_par[2, :, 13] = common_h

        deriv_par[2, :, 14] = kernel_h * common_h

        tanh_n = np.tanh((X[0, :]-par[15])/par[16])
        kernel_n = (1 - tanh_n * tanh_n)
        eta_n = 1 / 2 * (1 + tanh_n)
        tau_n = par[17] + par[18] * kernel_n
        common_n = (X[3, :] - eta_n) / (tau_n * tau_n)
        eta_der_n = - 1 / (2 * par[16]) * kernel_n
        tau_der_n = 2 * par[18] / par[16] * tanh_n * kernel_n
        deriv_par[3, :, 15] = eta_der_n / tau_n + tau_der_n * common_n

        eta_der_n = - (X[0, :] - par[15]) / (2 * (par[16] ** 2)) * kernel_n
        tau_der_n = 2 * par[18] * (X[0, :] - par[15]) / (par[16] ** 2) \
                    * tanh_n * kernel_n
        deriv_par[3, :, 16] = eta_der_n / tau_n + tau_der_n * common_n

        deriv_par[3, :, 17] = common_n

        deriv_par[3, :, 18] = kernel_n * common_n
        #===============================end here===============================
        return deriv_par

