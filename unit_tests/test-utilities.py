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

from pahmc_ode_cpu.utilities import Action  # import module to be tested
from pahmc_ode_cpu import lib_dynamics


class Test_utilities(unittest.TestCase):
    """Inherit the 'TestCase' module and build the test code below."""
    def test_action(self):
        """Unit test code below."""
        D = 20
        M = 10
        dt = np.random.rand()
        obsdim = list(set(np.random.randint(0, D, D//2)))
        Y = np.random.uniform(-1.0, 1.0, (len(obsdim),M))
        Rm = np.random.rand()
        Rf = np.random.rand() * 1e3
        stimuli = np.random.uniform(-1.0, 1.0, (D,M))

        #=========================type your code below=========================
        name = 'lorenz96'

        X = np.random.uniform(-8.0, 8.0, (D,M))
        par = np.array([8.17])  # this is for Lorenz96
        #===============================end here===============================
        try:
            dyn = getattr(lib_dynamics, f'Builtin_{name}')(name, stimuli)
        except:
            import def_dynamics
            dyn = def_dynamics.Dynamics(name, stimuli)

        A = Action(dyn, Y, dt, D, obsdim, M, Rm)
        fX = A.get_fX(X, par)

        compare_meas = 0
        for m in range(M):
            for a in range(D):
                if a in obsdim:
                    compare_meas = compare_meas \
                                   + (X[a, m] - Y[obsdim.index(a), m]) ** 2
        compare_meas = Rm / (2 * M) * compare_meas

        compare_model = 0
        for m in range(M-1):
            for a in range(D):
                compare_model = compare_model + (X[a, m+1] - fX[a, m]) ** 2
        compare_model = Rf / (2 * M) * compare_model

        compare = compare_meas + compare_model
        compare = np.around(compare, decimals=6)
        action = A.action(X, fX, Rf)
        action = np.around(action, decimals=6)

        self.assertEqual(action, compare)

    def test_dAdX(self):
        """Unit test code below."""
        D = 20
        M = 10
        dt = np.random.rand()
        obsdim = list(set(np.random.randint(0, D, D//2)))
        Y = np.random.uniform(-1.0, 1.0, (len(obsdim),M))
        Rm = np.random.rand()
        Rf = np.random.rand() * 1e3
        scaling = np.random.rand() * 1e3
        stimuli = np.random.uniform(-1.0, 1.0, (D,M))

        #=========================type your code below=========================
        name = 'lorenz96'

        X = np.random.uniform(-8.0, 8.0, (D,M))
        par = np.array([8.17])  # this is for Lorenz96
        #===============================end here===============================
        try:
            dyn = getattr(lib_dynamics, f'Builtin_{name}')(name, stimuli)
        except:
            import def_dynamics
            dyn = def_dynamics.Dynamics(name, stimuli)

        A = Action(dyn, Y, dt, D, obsdim, M, Rm)
        fX = A.get_fX(X, par)

        compare_meas = np.zeros((D,M))
        for a in range(D):
            for m in range(M):
                if a in obsdim:
                    compare_meas[a, m] = compare_meas[a, m] \
                                         + (X[a, m] - Y[obsdim.index(a), m])
                compare_meas[a, m] = Rm / M * compare_meas[a, m]

        compare_model = np.zeros((D,M))
        for a in range(D):
            for i in range(D):
                compare_model[a, 0] \
                  = compare_model[a, 0] \
                    + (X[i, 1] - fX[i, 0]) \
                      * ((i == a) + dt / 2 * dyn.jacobian(X, par)[i, a, 0])
            compare_model[a, 0] = - Rf / M * compare_model[a, 0]
            for i in range(D):
                compare_model[a, -1] \
                  = compare_model[a, -1] \
                    + (X[i, -1] - fX[i, -1]) \
                      * ((i == a) - dt / 2 * dyn.jacobian(X, par)[i, a, -1])
            compare_model[a, -1] = Rf / M * compare_model[a, -1]
            for m in range(1, M-1):
                part2 = 0
                for i in range(D):
                    part2 \
                      = part2 \
                        + (X[i, m] - fX[i, m-1]) \
                          * ((i == a) - dt / 2 * dyn.jacobian(X, par)[i, a, m])
                part2 = Rf / M * part2
                part3 = 0
                for i in range(D):
                    part3 \
                      = part3 \
                        + (X[i, m+1] - fX[i, m]) \
                          * ((i == a) + dt / 2 * dyn.jacobian(X, par)[i, a, m])
                part3 = - Rf / M * part3
                compare_model[a, m] = part2 + part3

        compare = scaling * (compare_meas + compare_model)
        compare = np.around(compare, decimals=6)
        dAdX = A.dAdX(X, par, fX, Rf, scaling)
        dAdX = np.around(dAdX, decimals=6)

        self.assertIs(np.array_equal(dAdX, compare), True)



    def test_dAdpar(self):
        """Unit test code below."""
        D = 20
        M = 10
        dt = np.random.rand()
        obsdim = list(set(np.random.randint(0, D, D//2)))
        Y = np.random.uniform(-1.0, 1.0, (len(obsdim),M))
        Rm = np.random.rand()
        Rf = np.random.rand() * 1e3
        scaling = np.random.rand() * 1e3
        stimuli = np.random.uniform(-1.0, 1.0, (D,M))

        #=========================type your code below=========================
        name = 'lorenz96'

        X = np.random.uniform(-8.0, 8.0, (D,M))
        par = np.array([8.17])  # this is for Lorenz96
        #===============================end here===============================
        try:
            dyn = getattr(lib_dynamics, f'Builtin_{name}')(name, stimuli)
        except:
            import def_dynamics
            dyn = def_dynamics.Dynamics(name, stimuli)

        A = Action(dyn, Y, dt, D, obsdim, M, Rm)
        fX = A.get_fX(X, par)

        compare = np.zeros(len(par))
        for b in range(len(par)):
            for i in range(D):
                for m in range(M-1):
                    compare[b] = compare[b] \
                                 + (X[i, m+1] - fX[i, m]) * dt / 2 \
                                   * (dyn.dfield_dpar(X, par)[i, m, b] \
                                     + dyn.dfield_dpar(X, par)[i, m+1, b])
            compare[b] = - Rf / M * compare[b]

        compare = scaling * compare
        compare = np.around(compare, decimals=6)
        dAdpar = A.dAdpar(X, par, fX, Rf, scaling)
        dAdpar = np.around(dAdpar, decimals=6)

        self.assertIs(np.array_equal(dAdpar, compare), True)


if __name__ == "__main__":
    unittest.main()

