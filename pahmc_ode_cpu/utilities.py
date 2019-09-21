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

        fX = np.zeros((self.D,self.M-1))
        for a in range(self.D):
            for m in range(self.M-1):
                fX[a, m] = X[a, m] + self.dt / 2 * (F[a, m+1] + F[a, m])

        return fX

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
        measerr = 0
        for m in range(self.M):
            for l in range(len(self.obsdim)):
                measerr = measerr + (X[self.obsdim[l], m] - self.Y[l, m]) ** 2
        measerr = self.Rm / (2 * self.M) * measerr

        modelerr = 0
        for m in range(self.M-1):
            for a in range(self.D):
                modelerr = modelerr + (X[a, m+1] - fX[a, m]) ** 2
        modelerr = Rf / (2 * self.M) * modelerr

        return measerr + modelerr

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
        J = self.dyn.jacobian(X, par)  # get the D-by-D-by-M Jacobian
        
        diff = np.zeros((self.D,self.M-1))
        for a in range(self.D):
            for m in range(self.M-1):
                diff[a, m] = X[a, m+1] - fX[a, m]

        part_meas = np.zeros((self.D,self.M))
        for m in range(self.M):
            for l in range(len(self.obsdim)):
                part_meas[self.obsdim[l], m] \
                  = self.Rm / self.M * (X[self.obsdim[l], m] - self.Y[l, m])

        part_model = np.zeros((self.D,self.M))
        for a in range(self.D):
            # m == 0 corner case
            for i in range(self.D):
                part_model[a, 0] = part_model[a, 0] + J[i, a, 0] * diff[i, 0]
            part_model[a, 0] \
              = - Rf / self.M * (diff[a, 0] + self.dt / 2 * part_model[a, 0])
            # m == M-1 corner case
            for i in range(self.D):
                part_model[a, -1] = part_model[a, -1] \
                                    + J[i, a, -1] * diff[i, -1]
            part_model[a, -1] \
              = Rf / self.M * (diff[a, -1] - self.dt / 2 * part_model[a, -1])
            # m == {1, ..., M-2}
            for m in range(1, self.M-1):
                for i in range(self.D):
                    part_model[a, m] = part_model[a, m] \
                                       + J[i, a, m] \
                                         * (diff[i, m-1] + diff[i, m])
                part_model[a, m] \
                  = Rf / self.M * (diff[a, m-1] - diff[a, m] \
                                   - self.dt / 2 * part_model[a, m])

        gradX_A = np.zeros((self.D,self.M))
        for a in range(self.D):
            for m in range(self.M):
                gradX_A[a, m] = scaling * (part_meas[a, m] + part_model[a, m])

        return gradX_A

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

        gradpar_A = np.zeros(len(par))
        for b in range(len(par)):
            for i in range(self.D):
                for m in range(self.M-1):
                    gradpar_A[b] = gradpar_A[b] \
                                   + (X[i, m+1] - fX[i, m]) * self.dt / 2 \
                                     * (G[i, m, b] + G[i, m+1, b])
            gradpar_A[b] = - scaling * Rf / self.M * gradpar_A[b]

        return gradpar_A

