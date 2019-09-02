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
'#=========================type your code below=========================
 #===============================end here===============================',
write down,
	1) The vector field of your dynamical system including external stimuli;
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

	def __init__(self, name, stimuli):
		"""
		This class defines the vector field.

		Inputs
		------
		   name: string specifying the name of the dynamics.
		stimuli: 2D array of external stimuli.
		"""
		self.name = name
		self.stimuli = stimuli

	def field(self, X, par, stimulus):
		"""
		This is your vector field.

		Parameters live in 'par'. You will later initialize 'par' in the main 
		script. You may choose an arbitrary order inside 'par', but you should 
		keep it consistent throughout.

		Inputs
		------
		       X: D-by-M numpy array for any positive integer M.
		     par: one-dimensional (shapeless) numpy array.
		stimulus: D-by-M numpy array for any positive integer M; stimulus is 
				  a subset of 'self.stimuli'.

		Returns
		-------
		vecfield: D-by-M numpy array for any positive integer M. 
				  Caution: make sure to include external stimulus, if any.
		"""
		(D, M) = np.shape(X)
		vecfield = np.zeros((D,M))  # initialize the output (with stimulus)

		#=========================type your code below=========================
		no need to change this line if using 'lib_dynamics'
		#===============================end here===============================
		return vecfield

	def jacobian(self, X, par):
		"""
		This is the Jacobian of your vector field.

		Inputs
		------
		  X: D-by-M numpy array for any positive integer M.
		par: one-dimensional (shapeless) numpy array.

		Returns
		-------
		jacob: D-by-D-by-M numpy array for any positive integer M.
		"""
		(D, M) = np.shape(X)
		idenmat = np.identity(D)
		jacob = np.zeros((D,D,M))  # initialize the output

		#=========================type your code below=========================
		no need to change this line if using 'lib_dynamics'
		#===============================end here===============================
		return jacob

	def dfield_dpar(self, X, par):
		"""
		This contains the derivatives of your vector field on the parameters.
		When constructing 'deriv_par', you should use the same order for the 
		output, 'deriv_par', as in 'par'.

		Inputs
		------
		  X: D-by-M numpy array for any positive integer M.
		par: one-dimensional (shapeless) numpy array.

		Returns
		-------
		deriv_par: D-by-M-by-len(par) numpy array. Each index in the third axis
				   corresponds to a D-by-M numpy array that contains the 
				   derivatives with respect to the path X.
		"""
		(D, M) = np.shape(X)
		deriv_par = np.zeros((D,M,len(par)))  # initialize the output

		#=========================type your code below=========================
		no need to change this line if using 'lib_dynamics'
		#===============================end here===============================
		return deriv_par
