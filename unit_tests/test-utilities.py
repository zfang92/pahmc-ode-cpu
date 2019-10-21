# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This is a unit test. If you would like to further develop pahmc_ode_cpu, you 
should visit here frequently. You should also be familiar with the Python (3.7)
built-in module 'unittest'.

To run this unit test, copy this file into its parent directory and run it.
"""


import numpy as np
import unittest

from pahmc_ode_cpu.__init__ import Fetch
from pahmc_ode_cpu import lib_dynamics


class Test_utilities(unittest.TestCase):
    """Inherit the 'TestCase' module and build the test code below."""
    def test_action(self):
        """Unit test code below."""
        D = np.int64(20)
        M = np.int64(10)
        dt = np.random.rand()
        obsdim \
          = np.array(list(set(np.random.randint(0, D, D//2))), dtype='int64')
        Y = np.random.uniform(-1.0, 1.0, (len(obsdim),M))
        Rm = np.random.rand()
        Rf = np.random.rand(D) * 1e3
        stimuli = np.random.uniform(-1.0, 1.0, (D,M))

        #=========================type your code below=========================
        name = 'lorenz96'

        X = np.random.uniform(-8.0, 8.0, (D,M))
        par = np.array([8.17])  # this is for Lorenz96
        #===============================end here===============================
        try:
            Fetch.Cls = getattr(lib_dynamics, f'Builtin_{name}')
        except:
            import def_dynamics
            Fetch.Cls = def_dynamics.Dynamics

        from pahmc_ode_cpu.utilities import Action

        dyn = (Fetch.Cls)(name, stimuli)

        # test get_fX()
        F = dyn.field(X, par, stimuli)
        compare_fX = X[:, :-1] + (F[:, 1:] + F[:, :-1]) * dt / 2

        A = Action(dyn, Y, dt, D, obsdim, M, Rm)
        fX = A.get_fX(X, par)

        self.assertIs(np.array_equal(fX, compare_fX), True)

        # test action()
        compare_meas = Rm / (2 * M) * np.sum((X[obsdim, :]-Y)**2)
        compare_model = np.sum(Rf/(2*M)*np.sum((X[:, 1:] - fX)**2, axis=1))

        compare = compare_meas + compare_model
        compare = np.around(compare, decimals=6)

        action = A.action(X, fX, Rf)
        action = np.around(action, decimals=6)

        self.assertEqual(action, compare)

    def test_dAdX(self):
        """Unit test code below."""
        D = np.int64(20)
        M = np.int64(10)
        dt = np.random.rand()
        obsdim \
          = np.array(list(set(np.random.randint(0, D, D//2))), dtype='int64')
        Y = np.random.uniform(-1.0, 1.0, (len(obsdim),M))
        Rm = np.random.rand()
        Rf = np.random.rand(D) * 1e3
        scaling = np.random.rand() * 1e3
        stimuli = np.random.uniform(-1.0, 1.0, (D,M))

        #=========================type your code below=========================
        name = 'lorenz96'

        X = np.random.uniform(-8.0, 8.0, (D,M))
        par = np.array([8.17])  # this is for Lorenz96
        #===============================end here===============================
        try:
            Fetch.Cls = getattr(lib_dynamics, f'Builtin_{name}')
        except:
            import def_dynamics
            Fetch.Cls = def_dynamics.Dynamics

        from pahmc_ode_cpu.utilities import Action

        dyn = (Fetch.Cls)(name, stimuli)

        A = Action(dyn, Y, dt, D, obsdim, M, Rm)
        fX = A.get_fX(X, par)

        # test dAdX()
        idenmat = np.zeros((D,D,1))
        idenmat[:, :, 0] = np.identity(D)

        J = dyn.jacobian(X, par)
        
        part1 = np.zeros((D,M))
        part1[obsdim, :] = Rm / M * (X[obsdim, :] - Y)

        kernel = np.zeros((D,1,M-1))
        kernel[:, 0, :] = Rf[:, np.newaxis] / M * (X[:, 1:] - fX)

        part2 = np.zeros((D,M))
        part2[:, 1:] = np.sum(kernel*(idenmat-dt/2*J[:, :, 1:]), 0)

        part3 = np.zeros((D,M))
        part3[:, :-1] = - np.sum(kernel*(idenmat+dt/2*J[:, :, :-1]), 0)

        compare = scaling * (part1 + part2 + part3)
        compare = np.around(compare, decimals=5)

        dAdX = A.dAdX(X, par, fX, Rf, scaling)
        dAdX = np.around(dAdX, decimals=5)

        self.assertIs(np.array_equal(dAdX, compare), True)



    def test_dAdpar(self):
        """Unit test code below."""
        D = np.int64(20)
        M = np.int64(10)
        dt = np.random.rand()
        obsdim \
          = np.array(list(set(np.random.randint(0, D, D//2))), dtype='int64')
        Y = np.random.uniform(-1.0, 1.0, (len(obsdim),M))
        Rm = np.random.rand()
        Rf = np.random.rand(D) * 1e3
        scaling = np.random.rand() * 1e3
        stimuli = np.random.uniform(-1.0, 1.0, (D,M))

        #=========================type your code below=========================
        name = 'lorenz96'

        X = np.random.uniform(-8.0, 8.0, (D,M))
        par = np.array([8.17])  # this is for Lorenz96
        #===============================end here===============================
        try:
            Fetch.Cls = getattr(lib_dynamics, f'Builtin_{name}')
        except:
            import def_dynamics
            Fetch.Cls = def_dynamics.Dynamics

        from pahmc_ode_cpu.utilities import Action

        dyn = (Fetch.Cls)(name, stimuli)

        A = Action(dyn, Y, dt, D, obsdim, M, Rm)
        fX = A.get_fX(X, par)

        # test dAdpar
        G = dyn.dfield_dpar(X, par)  # get the D-by-M-by-len(par) array

        kernel = np.zeros((D,M-1,1))
        kernel[:, :, 0] = Rf[:, np.newaxis] / M * (X[:, 1:] - fX)
        kernel = kernel * dt / 2 * (G[:, :-1, :] + G[:, 1:, :])

        compare = - scaling * np.sum(np.sum(kernel, 0), 0)
        compare = np.around(compare, decimals=6)

        dAdpar = A.dAdpar(X, par, fX, Rf, scaling)
        dAdpar = np.around(dAdpar, decimals=6)

        self.assertIs(np.array_equal(dAdpar, compare), True)


if __name__ == "__main__":
    unittest.main()

