# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This contains necessary functions in order to run PAHMC. The functions are 
to be called by method 'hmc' in 'pahmc.py'.
"""


import numpy as np


class Action:
	"""
	This class contains useful functions to evaluate the action and its 
	derivatives with respect to the state variables and the parameters.
	"""

	def __init__(self):
		"""This class is to be instantiated internally in 'pahmc.py'."""

	def get_fX(self, dyn, dt, X, par):
		"""
		This method calculates the discretized vector field (lowercase f in the
		paper). The discretization rule is trapezoidal.

		Inputs
		------
		dyn: an object instantiated using 'def_dynamics.Dynamics'.
		 dt: discretization interval.
		  X: the state variable with shape (D, M).
		par: the parameters as a Python (3.7 or later) dictionary.

		Returns
		-------
		fX: the discretized vector field with shape (D, M). Each column 
			corresponds to the vector field at a given time.
		"""
		fX = X + (dyn.field(np.roll(X, -1, 1), par) + dyn.field(X, par)) \
				 * dt / 2

		return fX

	def action(self, dyn, X, fX, obsdim, M, Y, Rm, Rf):
		"""
		This method calculates the action.

		Inputs
		------
		   dyn: an object instantiated using 'def_dynamics.Dynamics'.
		     X: the state variable with shape (D, M).
		    fX: the discretized vector field with shape (D, M). Each column 
				corresponds to the vector field at a given time.
		obsdim: Python list containing the observed dimensions chosen 
		        from the set {1, ..., D}.
		     M: number of time steps actually being used to train the 
		        model.
		     Y: the training data (a "shorter" version of data_noisy).
		    Rm: scalar.
		    Rf: numpy array of length betamax.

		Returns
		-------
		the action. See the paper for its form.
		"""
		measuerr = X[obsdim, :] - Y
		measuerr = Rm / (2 * M) * np.sum(measuerr*measuerr)
		modelerr = (np.roll(X, -1, 1) - fX)[:, 0:M-1]
		modelerr = Rf / (2 * M) * np.sum(modelerr*modelerr)

		return measuerr + modelerr

	def dAdX(self, dyn, dt, X, fX, D, obsdim, M, Y, Rm, Rf, scaling):
		"""
		This method calculates the derivatives of the action with respect to 
		the path X.

		Inputs
		------
			dyn: an object instantiated using 'def_dynamics.Dynamics'.
			 dt: discretization interval.
			  X: the state variable with shape (D, M).
			 fX: the discretized vector field with shape (D, M). Each column 
				 corresponds to the vector field at a given time.
			  D: model degrees of freedom.
		 obsdim: Python list containing the observed dimensions chosen 
		         from the set {1, ..., D}.
			  M: number of time steps actually being used to train the 
		         model.
			  Y: the training data (a "shorter" version of data_noisy).
			 Rm: scalar.
			 Rf: numpy array of length betamax.
		scaling: float or a Python list of length betamax. When a float 
				 input is given, scaling will be broadcasted into a list 
				 of length betamax; when a list is given, scaling remains 
				 itself.

		Returns
		-------
		the dirivatives of the action with respect to the path.
		"""
		a
