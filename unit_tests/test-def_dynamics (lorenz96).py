# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This is a unit test. If you would like to further develop pahmc_ode_cpu, you 
should visit here frequently. You should also be familiar with the Python (3.7)
built-in module 'unittest'.

To run this unit test, copy this file into its parent directory and eXecute it.
"""


import numpy as np
import unittest

from def_dynamics import Dynamics  # import module to be tested


class Test_def_dynamics(unittest.TestCase):
	"""Inherit the 'TestCase' module and build the test code below"""
	def test_field(self):
		#===============type your code below===============
		D = 20
		M = 10

		X = np.random.uniform(-8.0, 8.0, (D,M))
		par = {'nu':8.17}
		compare = np.zeros((D, M))

		for m in range(M):
			compare[0, m] = (X[1, m] - X[D-2, m]) * X[D-1, m] - X[0, m]
			compare[1, m] = (X[2, m] - X[D-1, m]) * X[0, m] - X[1, m]
			compare[D-1, m] = (X[0, m] - X[D-3, m]) * X[D-2, m] - X[D-1, m]
			for a in range(2, D-1):
				compare[a, m] = (X[a+1, m] - X[a-2, m]) * X[a-1, m] - X[a, m]
		compare = compare + par['nu']

		vecfield = Dynamics('testdyn').field(X, par)
		#=====================end here=====================
		self.assertEqual(vecfield.shape, (D,M))
		self.assertIs(np.array_equal(vecfield, compare), True)

	def test_jacobian(self):
		#===============type your code below===============
		D = 20
		M = 10

		X = np.random.uniform(-8.0, 8.0, (D,M))
		par = {'nu':8.17}
		compare = np.zeros((D,D,M))

		def ind(i, D): return 1 + (i - 1) % D

		for m in range(M):
			for i in range(1, D+1):
				for j in range(1, D+1):
					compare[i-1, j-1, m] \
						= (ind(i-1, D) == j) \
					  	  * (X[ind(i+1, D)-1, m] - X[ind(i-2, D)-1, m]) \
				      	  + ((ind(i+1, D) == j) - (ind(i-2, D) == j)) \
				      	  * X[ind(i-1, D)-1, m] - (i == j)

		jacob = Dynamics('testdyn').jacobian(X, par)
		#=====================end here=====================
		self.assertEqual(jacob.shape, (D,D,M))
		self.assertIs(np.array_equal(jacob, compare), True)

	def test_dfield_dpar(self):
		return


if __name__ == "__main__":
	unittest.main()
