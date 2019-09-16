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
  
from pahmc_ode_cpu import lib_dynamics


class Test_def_dynamics(unittest.TestCase):
    """Inherit the 'TestCase' module and build the test code below."""
    def test_field(self):
        #=========================type your code below=========================
        name = 'lorenz96'

        D = 20
        M = 10
        X = np.random.uniform(-8.0, 8.0, (D,M))
        par = np.array([8.17])

        stimuli = np.random.uniform(-1.0, 1.0, (D,M))

        compare = (np.roll(X, -1, 0) - np.roll(X, 2, 0)) * np.roll(X, 1, 0) \
                  - X + par[0]
        #===============================end here===============================
        try:
            dyn = getattr(lib_dynamics, f'Builtin_{name}')(name, stimuli)
        except:
            import def_dynamics
            dyn = def_dynamics.Dynamics(name, stimuli)

        vecfield = dyn.field(X, par, stimuli)
        vecfield_shift1 = dyn.field(np.roll(X, -2, 1), par, 
                                    np.roll(stimuli, -2, 1))
        vecfield_shift2 = np.roll(vecfield, -2, 1)

        self.assertEqual(np.shape(vecfield), (D,M))
        self.assertIs(np.array_equal(vecfield, compare), True)
        self.assertIs(np.array_equal(vecfield_shift1, vecfield_shift2), True)

    def test_jacobian(self):
        #=========================type your code below=========================
        name = 'lorenz96'

        D = 20
        M = 10
        X = np.random.uniform(-8.0, 8.0, (D,M))
        par = np.array([8.17])

        stimuli = np.random.uniform(-1.0, 1.0, (D,M))

        idenmat = np.identity(D)

        compare = np.roll(idenmat, -1, 1)[:, :, np.newaxis] \
                  * np.reshape((np.roll(X, -1, 0)-np.roll(X, 2, 0)), (D,1,M)) \
                  + (np.roll(idenmat, 1, 1)\
                     -np.roll(idenmat, -2, 1))[:, :, np.newaxis] \
                  * np.reshape(np.roll(X, 1, 0), (D,1,M)) \
                  - idenmat[:, :, np.newaxis]
        #===============================end here===============================
        try:
            dyn = getattr(lib_dynamics, f'Builtin_{name}')(name, stimuli)
        except:
            import def_dynamics
            dyn = def_dynamics.Dynamics(name, stimuli)

        jacob = dyn.jacobian(X, par)
        jacob_shift1 = dyn.jacobian(np.roll(X, -2, 1), par)
        jacob_shift2 = np.roll(jacob, -2, 2)

        self.assertEqual(np.shape(jacob), (D,D,M))
        self.assertIs(np.array_equal(jacob, compare), True)
        self.assertIs(np.array_equal(jacob_shift1, jacob_shift2), True)

    def test_dfield_dpar(self):
        #=========================type your code below=========================
        name = 'lorenz96'

        D = 20
        M = 10
        X = np.random.uniform(-8.0, 8.0, (D,M))
        par = np.array([8.17])

        stimuli = np.random.uniform(-1.0, 1.0, (D,M))

        compare = np.ones((D,M,1))
        #===============================end here===============================
        try:
            dyn = getattr(lib_dynamics, f'Builtin_{name}')(name, stimuli)
        except:
            import def_dynamics
            dyn = def_dynamics.Dynamics(name, stimuli)

        deriv_par = dyn.dfield_dpar(X, par)

        self.assertIs(np.array_equal(deriv_par, compare), True)


if __name__ == "__main__":
    unittest.main()

