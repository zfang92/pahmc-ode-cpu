# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This module generates twin-experiemnt (training) data for pahmc_ode_cpu with 
the specs provided by the user.
"""


import numpy as np
from pathlib import Path


class Data:
    """
    This is where the twin-experiment data are generated using the specs 
    provided by the user.
    """

    def __init__(self):
        """The name of the dynamics is included below in 'dyn'."""

    def generate(self, dyn, D, length, dt, noise, par_true, x0, burndata):
        """
        This method first searches for an existing data file by looking for a
        data filename that matches the name of the user-defined dynamics. If 
        found successfully, it then load the file and compare the detailed 
        specs of its data with user specs; if everything matches, 'generate' 
        then output the existing data. In all other cases, it integrates the 
        user-defined dynamics using a trapezoidal rule and then outputs the 
        generated data and saves the data files.

        Inputs
        ------
             dyn: the dynamics.
               D: model degrees of freedom.
          length: number of discrete time steps of the generated data.
              dt: discretization interval.
           noise: 1d (shapeless) numpy array for the standard deviations.
        par_true: true parameters used to generate the data.
              x0: 1d (shapeless) numpy array of the initial condition.
        burndata: Boolean for burning the first half of the generated data.

        Returns
        -------
        data_noisy: D-by-length numpy array of the generated noisy data.
        """
        print('\nGenerating data... ', end='')

        if burndata == True:
            start = length
        else:
            start = 0

        if np.shape(par_true) == ():
            par_true = np.array([par_true])

        filepath = Path.cwd() / 'user_data'

        if (filepath / f'{dyn.name}.npz').exists():  # if a match is found
            noisyfile = np.load(filepath/f'{dyn.name}.npz')

            if (np.shape(noisyfile['data']) == (D, length)) \
            and (noisyfile['dt'] == dt) \
            and np.array_equal(noisyfile['noise'], noise) \
            and np.array_equal(noisyfile['par_true'], par_true) \
            and (bool(noisyfile['burndata']) == burndata) \
            and np.array_equal(noisyfile['stimuli'], 
                               dyn.stimuli[:, start:start+length]):
                data_noisy = noisyfile['data']
                noisyfile.close()

                print('successful (data with the same specs already exist).\n')
                return data_noisy, dyn.stimuli[:, start:start+length]

        # for all other cases
        rawdata = np.zeros((D,start+length))
        rawdata[:, 0] = x0
        
        for k in range(start+length-1):
            # Newton-Raphson's initial guess using the Euler method
            x_start \
              = rawdata[:, [k]] + dt * dyn.field(rawdata[:, [k]], par_true, 
                                                 dyn.stimuli[:, [k]])

            # first iteration of Newton-Raphson for the trapezoidal rule
            g_x = dt / 2 * (dyn.field(x_start, par_true, 
                                      dyn.stimuli[:, [k+1]])[:, 0] \
                            + dyn.field(rawdata[:, [k]], par_true, 
                                        dyn.stimuli[:, [k]])[:, 0]) \
                  + rawdata[:, k] - x_start[:, 0]
            J = dt / 2 * dyn.jacobian(x_start, par_true)[:, :, 0] \
                - np.identity(D)

            x_change = np.linalg.solve(J, g_x)[:, np.newaxis]
            x_new = x_start - x_change
            x_start = x_new

            # iterate until the correction reaches tolerance level
            while np.sum(abs(x_change)) > 1e-13:
                g_x = dt / 2 * (dyn.field(x_start, par_true, 
                                          dyn.stimuli[:, [k+1]])[:, 0] \
                                + dyn.field(rawdata[:, [k]], par_true, 
                                            dyn.stimuli[:, [k]])[:, 0]) \
                      + rawdata[:, k] - x_start[:, 0]
                J = dt / 2 * dyn.jacobian(x_start, par_true)[:, :, 0] \
                    - np.identity(D)

                x_change = np.linalg.solve(J, g_x)[:, np.newaxis]
                x_new = x_start - x_change
                x_start = x_new

            rawdata[:, [k+1]] = x_new  # final value
        
        data_noiseless = rawdata[:, start:start+length]
        np.savez(filepath/f'{dyn.name}_noiseless', 
                 data=data_noiseless, 
                 dt=dt, 
                 noise=np.zeros(D), 
                 par_true=par_true, 
                 burndata=burndata, 
                 stimuli=dyn.stimuli[:, start:start+length])

        data_noisy = np.zeros((D,length))
        for a in range(D):
            data_noisy[a, :] \
              = data_noiseless[a, :] + np.random.normal(0, noise[a], length)
        np.savez(filepath/f'{dyn.name}', 
                 data=data_noisy, 
                 dt=dt, 
                 noise=noise, 
                 par_true=par_true, 
                 burndata=burndata, 
                 stimuli=dyn.stimuli[:, start:start+length])

        print('successful.\n')
        return data_noisy, dyn.stimuli[:, start:start+length]

