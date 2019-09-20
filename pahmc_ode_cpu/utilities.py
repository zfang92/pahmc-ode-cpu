# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This contains necessary functions in order to run PAHMC. The functions are 
to be called by method 'hmc' in 'pahmc.py'.
"""


from numba import jitclass, types
import numpy as np

from pahmc_ode_cpu.__init__ import Fetch


dyn_spec = types.deferred_type()
dyn_spec.define(Fetch.Cls.class_type.instance_type)

spec = [('dyn', dyn_spec), 
        ('Y', types.float64[:, :]), 
        ('dt', types.float64), 
        ('D', types.int64), 
        ('obsdim', types.int64[:]), 
        ('M', types.int64), 
        ('Rm', types.float64)]
@jitclass(spec)
class Action:
    """
    This class contains useful functions to evaluate the action and its 
    derivatives with respect to the state variables and the parameters.
    """

    def __init__(self, dyn, Y, dt, D, obsdim, M, Rm):
        """
        This class is to be instantiated internally in 'pahmc.py'.

        Inputs
        ------
           dyn: an object instantiated using 'def_dynamics.Dynamics'.
             Y: the training data (a "shorter" version of data_noisy).
            dt: discretization interval.
             D: model degrees of freedom.
        obsdim: 1d (shapeless) numpy array of integers.
             M: number of time steps actually being used to train the model.
            Rm: scalar.
        """
        self.dyn = dyn
        self.Y = Y
        self.dt = dt
        self.D = D
        self.obsdim = obsdim
        self.M = M
        self.Rm = Rm

    def get_fX(self, X, par):
        """
        This method calculates the discretized vector field (lowercase f in the
        paper). The discretization rule is trapezoidal.

        Inputs
        ------
          X: the state variable with shape (D, M).
        par: one-dimensional (shapeless) numpy array.

        Returns
        -------
        the discretized vector field with shape (D, M-1). Each column 
        corresponds to the vector field at a given time.
        """
        # get the original vector field
        F = self.dyn.field(X, par, self.dyn.stimuli)

        return X[:, :-1] + (F[:, 1:] + F[:, :-1]) * self.dt / 2

    def action(self, X, fX, Rf):
        """
        This method calculates the action.

        Inputs
        ------
         X: the state variable with shape (D, M).
        fX: the discretized vector field with shape (D, M-1). Each column 
            corresponds to the vector field at a given time.
        Rf: numpy array of length betamax.

        Returns
        -------
        the action. See the paper for its form.
        """
        measuerr = np.zeros((len(self.obsdim),self.M))
        for l in range(len(self.obsdim)):
            measuerr[l, :] = X[self.obsdim[l], :] - self.Y[l, :]
        measuerr = self.Rm / (2 * self.M) * np.sum(measuerr*measuerr)

        modelerr = X[:, 1:] - fX
        modelerr = Rf / (2 * self.M) * np.sum(modelerr*modelerr)

        return measuerr + modelerr

    def dAdX(self, X, par, fX, Rf, scaling):
        """
        This method calculates the derivatives of the action with respect to 
        the path X.

        Inputs
        ------
              X: the state variable with shape (D, M).
            par: one-dimensional (shapeless) numpy array.
             fX: the discretized vector field with shape (D, M-1). Each column 
                 corresponds to the vector field at a given time.
             Rf: numpy array of length betamax.
        scaling: 1d (shapeless) numpy array of floats, with length betamax.

        Returns
        -------
        D-by-M numpy array that contains the dirivatives of the action with 
        respect to the path X.
        """
        idenmat = np.zeros((self.D,self.D,1))
        idenmat[:, :, 0] = np.identity(self.D)

        J = self.dyn.jacobian(X, par)  # get the D-by-D-by-M Jacobian
        
        part1 = np.zeros((self.D,self.M))
        for l in range(len(self.obsdim)):
            part1[self.obsdim[l], :] = X[self.obsdim[l], :] - self.Y[l, :]
        part1 = self.Rm / self.M * part1

        kernel = np.zeros((self.D,1,self.M-1))
        kernel[:, 0, :] = X[:, 1:] - fX

        part2 = np.zeros((self.D,self.M))
        part2[:, 1:] \
          = Rf / self.M * np.sum(kernel*(idenmat-self.dt/2*J[:, :, 1:]), 0)

        part3 = np.zeros((self.D,self.M))
        part3[:, :-1] \
          = - Rf / self.M * np.sum(kernel*(idenmat+self.dt/2*J[:, :, :-1]), 0)

        return scaling * (part1 + part2 + part3)

    def dAdpar(self, X, par, fX, Rf, scaling):
        """
        This method calculates the derivatives of the action with respect to 
        the parameters 'par'.

        Inputs
        ------
              X: the state variable with shape (D, M).
            par: one-dimensional (shapeless) numpy array.
             fX: the discretized vector field with shape (D, M-1). Each column 
                 corresponds to the vector field at a given time.
             Rf: numpy array of length betamax.
        scaling: 1d (shapeless) numpy array of floats, with length betamax.

        Returns
        -------
        one-dimensional (shapeless) numpy array of length len(par).
        """
        G = self.dyn.dfield_dpar(X, par)  # get the D-by-M-by-len(par) array

        kernel = np.zeros((self.D,self.M-1,1))
        kernel[:, :, 0] = X[:, 1:] - fX
        kernel = kernel * self.dt / 2 * (G[:, :-1, :] + G[:, 1:, :])

        return scaling * (- Rf / self.M * np.sum(np.sum(kernel, 0), 0))

