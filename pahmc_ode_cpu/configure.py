# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

The purpose of this module is three-fold:
    1) error/exception handling---check the shape, type, etc. of all the user-
    provided variables and make sure they meet the requirements;
    2) regulation---make every variable have a fixed type and shape after 
    receiving from the user, then save these varialbles as a .npz file;
    3) reading from the stimuli file, if any.
"""


from pathlib import Path

import numpy as np


class Configure:
    """Check variables, do the regulation, get the time-dependent stimuli."""

    def __init__(self, name, 
                 D, M, obsdim, dt, 
                 Rf0, alpha, betamax, 
                 n_iter, epsilon, S, mass, scaling, 
                 soft_dynrange, par_start, 
                 length, noise, par_true, x0, burndata):
        self.name = name
        self.D = D
        self.M = M
        self.obsdim = obsdim
        self.dt = dt
        self.Rf0 = Rf0
        self.alpha = alpha
        self.betamax = betamax
        self.n_iter = n_iter
        self.epsilon = epsilon
        self.S = S
        self.mass = mass
        self.scaling = scaling
        self.soft_dynrange = soft_dynrange
        self.par_start = par_start
        self.length = length
        self.noise = noise
        self.par_true = par_true
        self.x0 = x0
        self.burndata = burndata

    def check_all(self):
        """
        Check everything.

        The error messages below pretty much tell what is required for each 
        intput variable.
        """
        assert type(self.name) == str, \
            "'name' must be a string."

        assert type(self.D) == int, \
            "'D' must be a plain integer."

        assert (type(self.M) == int \
                and self.M >= 3), \
            "'M' msut be a plain integer greater than 2."

        assert (type(self.obsdim) == list \
                and len(np.shape(self.obsdim)) == 1 \
                and self.obsdim == list(np.floor(self.obsdim)) \
                and self.obsdim == list(set(self.obsdim))), \
            "'obsdim' must be a one-dimensional list" \
            + " that contains only plain integers;" \
            + " it must also be in ascending order with no repetition inside."

        assert type(self.dt) == float, \
            "'dt' must be a plain float."

        assert type(self.Rf0) == float, \
            "'Rf0' must be a plain float."

        assert type(self.alpha) == float, \
            "'alpha' must be a plain float."

        assert (type(self.betamax) == int \
                and self.betamax >= 1), \
            "'betamax' must be a plain integer greater than 0."

        assert ((type(self.n_iter) == int \
                 and np.shape(self.n_iter) == ()) \
                or \
                (type(self.n_iter) == np.ndarray \
                 and np.shape(self.n_iter) == (self.betamax, ) \
                 and np.array_equal(self.n_iter, np.floor(self.n_iter)))), \
            "'n_iter' must be either a plain integer or a one-dimensional" \
            + " array of length 'betamax' that contains only plain integers."

        assert ((type(self.epsilon) == float \
                 and np.shape(self.epsilon) == ()) \
                or \
                (type(self.epsilon) == np.ndarray \
                 and np.shape(self.epsilon) == (self.betamax, ))), \
            "'epsilon' must be either a plain float or a one-dimensional" \
            + " array of length 'betamax' that contains only plain floats."

        assert ((type(self.S) == int \
                 and np.shape(self.S) == ()) \
                or \
                (type(self.S) == np.ndarray \
                 and np.shape(self.S) == (self.betamax, ) \
                 and np.array_equal(self.S, np.floor(self.S)))), \
            "'S' must be either a plain integer or a one-dimensional" \
            + " array of length 'betamax' that contains only plain integers."

        assert ((type(self.mass) == tuple \
                 and np.shape(self.mass) == (3, )) \
                or \
                (type(self.mass) == np.ndarray \
                 and np.shape(self.mass) == (self.betamax, 3))), \
            "'mass' must be either a 3-tuple or an array of shape" \
            + " (betamax, 3)."

        assert ((type(self.scaling) == float \
                 and np.shape(self.scaling) == ()) \
                or \
                (type(self.scaling) == np.ndarray \
                 and np.shape(self.scaling) == (self.betamax, ))), \
            "'scaling' must be either a plain float or a one-dimensional" \
            + " array of length 'betamax' that contains only plain floats."

        assert ((type(self.soft_dynrange) == tuple \
                 and np.shape(self.soft_dynrange) == (2, )) \
                or \
                (type(self.soft_dynrange) == np.ndarray \
                 and np.shape(self.soft_dynrange) == (self.D, 2))), \
            "'soft_dynrange' must be either a 2-tuple or an array of shape" \
            + " (D, 2)."

        assert ((type(self.par_start) == float \
                 and np.shape(self.par_start) == ()) \
                or \
                (type(self.par_start) == np.ndarray \
                 and len(np.shape(self.par_start)) == 1)), \
            "'par_start' must be either a plain float or a one-dimensional" \
            + " array that contains only plain floats."

        assert (type(self.length) == int \
                and self.length > self.M), \
            "'length' must be a plain integer greater than 'M'."

        assert np.shape(self.noise) == (self.D, ), \
            "'noise' must be a one-dimensional array with length 'D'."

        assert ((type(self.par_true) == float \
                 and np.shape(self.par_true) == ()) \
                or \
                (type(self.par_true) == np.ndarray \
                 and len(np.shape(self.par_true)) == 1)), \
            "'par_true' must be either a plain float or a one-dimensional" \
            + " array that contains only plain floats."

        assert np.shape(self.x0) == (self.D, ), \
            "'x0' must be a one-dimensional array with length 'D'."

        assert type(self.burndata) == bool, \
            "'burndata' must be a Boolean."

    def regulate(self):
        """
        Regulate the type and shape for each input variable.

        The code itself pretty much tells what each variable becomes after 
        being regulated.
        """
        self.name = str(self.name)

        self.D = np.int64(self.D)

        self.M = np.int64(self.M)

        self.obsdim = np.array(self.obsdim, dtype='int64') - 1

        self.dt = float(self.dt)

        self.Rf0 = float(self.Rf0)

        self.alpha = float(self.alpha)

        self.betamax = np.int64(self.betamax)

        self.n_iter = np.ones(self.betamax, dtype='int64') \
                      * np.array(self.n_iter, dtype='int64')

        self.epsilon = np.ones(self.betamax, dtype='float64') \
                       * np.array(self.epsilon, dtype='float64')

        self.S = np.ones(self.betamax, dtype='int64') \
                 * np.array(self.S, dtype='int64')

        self.mass = np.ones((self.betamax,3), dtype='float64') \
                    * np.array(self.mass, dtype='float64')

        self.scaling = np.ones(self.betamax, dtype='float64') \
                       * np.array(self.scaling, dtype='float64')

        self.soft_dynrange = np.ones((self.D,2), dtype='float64') \
                             * np.array(self.soft_dynrange, dtype='float64')

        if np.shape(self.par_start) == ():
            self.par_start = np.array([self.par_start], dtype='float64')
        else:
            self.par_start = np.array(self.par_start, dtype='float64')

        self.length = np.int64(self.length)

        self.noise = np.array(self.noise, dtype='float64')

        if np.shape(self.par_true) == ():
            self.par_true = np.array([self.par_true], dtype='float64')
        else:
            self.par_true = np.array(self.par_true, dtype='float64')

        self.x0 = np.array(self.x0, dtype='float64')

        self.burndata = bool(self.burndata)

        np.savez(Path.cwd()/'user_results'/'config',
                 name=self.name, 
                 D=self.D, 
                 M=self.M, 
                 obsdim=self.obsdim, 
                 dt=self.dt, 
                 Rf0=self.Rf0, 
                 alpha=self.alpha, 
                 betamax=self.betamax, 
                 n_iter=self.n_iter, 
                 epsilon=self.epsilon, 
                 S=self.S, 
                 mass=self.mass, 
                 scaling=self.scaling, 
                 soft_dynrange=self.soft_dynrange, 
                 par_start=self.par_start, 
                 length=self.length, 
                 noise=self.noise, 
                 par_true=self.par_true, 
                 x0=self.x0, 
                 burndata=self.burndata)

        return self.name, \
               self.D, self.M, self.obsdim, self.dt, \
               self.Rf0, self.alpha, self.betamax, \
               self.n_iter, self.epsilon, self.S, self.mass, self.scaling, \
               self.soft_dynrange, self.par_start, \
               self.length, self.noise, self.par_true, self.x0, self.burndata

    def get_stimuli(self):
        """
        First look for the stimuli file; if found, check, regulate, and load. 
        If not found, set to default and load.
        """
        filename = Path.cwd() / 'user_data' / f'{self.name}_stimuli.npy'

        if filename.exists():
            stimuli = np.load(filename)

            assert (stimuli.dtype == 'float64' \
                    and len(np.shape(stimuli)) == 2 \
                    and np.shape(stimuli)[0] == self.D \
                    and np.shape(stimuli)[1] >= 2 * self.length), \
                "The stimuli provided must be a numpy array of pure floats," \
                + " and the array must have shape (D, #) with # >= 2*'length'."

            print('\nFound external stimuli in '+f'{self.name}_stimuli.npy.')
            return stimuli[:, :2*self.length]
        else:
            print('\nNo external stimuli provided, setting it to zero...')
            return np.zeros((self.D,2*self.length), dtype='float64')

