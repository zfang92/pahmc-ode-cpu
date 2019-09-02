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


import numpy as np


class Builtin_lorenz96:
    """
    This class implements the standard Lorenz96 model. Fortunately, there is
    only one representation of the model.
    """

    def __init__(self, name, stimuli):
        self.name = name
        self.stimuli = stimuli

    def field(self, X, par, stimulus):
        return (np.roll(X, -1, 0) - np.roll(X, 2, 0)) * np.roll(X, 1, 0) \
               - X + par[0] + stimulus

    def jacobian(self, X, par):
        (D, M) = np.shape(X)
        idenmat = np.identity(D)

        jacob = np.roll(idenmat, -1, 1)[:, :, np.newaxis] \
                * np.reshape((np.roll(X, -1, 0)-np.roll(X, 2, 0)), (D,1,M)) \
                + (np.roll(idenmat, 1, 1)\
                   -np.roll(idenmat, -2, 1))[:, :, np.newaxis] \
                * np.reshape(np.roll(X, 1, 0), (D,1,M)) \
                - idenmat[:, :, np.newaxis]
        
        return jacob

    def dfield_dpar(self, X, par):
        (D, M) = np.shape(X)

        return np.ones((D,M,len(par)))

