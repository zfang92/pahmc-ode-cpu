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
'#===============type your code below===============
 #=====================end here=====================',
write down,
	1) The vector field of your dynamical system;
	2) Its Jacobian;
	3) The derivatives with respect to each parameter of the system.

Double check, and pay special attention to the shape requirements for the 
intputs and outputs. Mistakes made here are not easily detectable by looking at 
the outputs. When in doubt, check with the user manual. You can use the 
corresponding file in 'unit_tests' or write your own unit test module if you 
prefer debugging this way.
"""


import numpy as np


class Dynamics:
	"""
	This should contain all that pahmc_ode_cpu needs to know about your 
	dynamical system.
	"""

	def __init__(self, name):
		"""The name will be the unique identifier to a dynamical system"""
		self.name = name

	def field(self, X, par):
		"""
		This is your vector field.

		Parameters live in 'par'. You will later initialize 'par' in the main 
		script. You may choose an arbitrary order inside 'par', but you should 
		keep it consistent throughout.

		Inputs
		------
		  X: D-by-M numpy array for M > 1. If M == 1, X is a one-dimensional 
		     (shapeless) numpy array.
		par: one-dimensional (shapeless) numpy array.

		Returns
		-------
		vecfield: D-by-M numpy array for any positive integer M.
		"""
		if len(X.shape) == 1:  # if M == 1, X.shape will be equal to D
			X = X[:, np.newaxis]  # in this case, make X.shape equal to (D, 1)

		vecfield = np.zeros(X.shape)  # initialize the output

		#===============type your code below===============
		vecfield = (np.roll(X, -1, 0) - np.roll(X, 2, 0)) \
			* np.roll(X, 1, 0) - X + par[0]
		#=====================end here=====================
		return vecfield

	def jacobian(self, X, par):
		"""
		This is the Jacobian of your vector field.

		Inputs
		------
		  X: D-by-M numpy array for M > 1. If M == 1, X is a one-dimensional 
		     (shapeless) numpy array.
		par: one-dimensional (shapeless) numpy array.

		Returns
		-------
		jacob: D-by-D-by-M numpy array for any positive integer M.
		"""
		if len(X.shape) == 1:  # if M == 1, X.shape will be equal to D
			X = X[:, np.newaxis]  # in this case, make X.shape equal to (D, 1)
		shape = X.shape

		idenmat = np.identity(shape[0], dtype='float64')
		jacob = np.zeros((shape[0],shape[0],shape[1]))  # initialization

		#===============type your code below===============
		jacob = np.roll(idenmat, -1, 1)[:, :, np.newaxis] \
				* np.reshape((np.roll(X, -1, 0)-np.roll(X, 2, 0)), 
					(shape[0],1,shape[1])) \
				+ (np.roll(idenmat, 1, 1)\
					-np.roll(idenmat, -2, 1))[:, :, np.newaxis] \
				* np.reshape(np.roll(X, 1, 0), (shape[0],1,shape[1])) \
				- idenmat[:, :, np.newaxis]
		#=====================end here=====================
		return jacob

	def dfield_dpar(self, X, par):
		"""
		This contains the derivatives of your vector field on the parameters.
		When constructing 'deriv_par', you should use the same order for the 
		output, 'deriv_par', as in 'par'.

		Inputs
		------
		  X: D-by-M numpy array for M > 1. If M == 1, X is a one-dimensional 
		     (shapeless) numpy array.
		par: one-dimensional (shapeless) numpy array.

		Returns
		-------
		deriv_par: D-by-M-by-len(par) numpy array. Each index in the third axis
				   corresponds to a D-by-M numpy array that contains the 
				   derivatives with respect to the path X.
		"""
		if len(X.shape) == 1:  # if M == 1, X.shape will be equal to D
			X = X[:, np.newaxis]  # in this case, make X.shape equal to (D, 1)
		
		deriv_par = np.zeros(D,M,len(par))

		#===============type your code below===============
		deriv_par[:, :, 0] = np.ones(X.shape)
		#=====================end here=====================
		return deriv_par
