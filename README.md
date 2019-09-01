# pahmc-ode-cpu

A user-manual is forthcoming. Meanwhile, the docstrings within the code should be self-explanatory enough to get you started.

Python version 3.6.x or higher is required.

Before running the code, the user is assumed to have the following:
	1) The dynamical system. If calling one of the built-in examples, the name
	of the dynamics must have a match in 'lib_dynamics.py'; if builing from 
	scratch, 'def_dynamics.py' must be ready to go.
	2) The data. If performing twin-experiments, the specs should be given but
	a data file is not required; if working with real data, the data should be
  prepared according to the user manual.
	3) If external stimuli are needed, a .npy file containing the time series; 
	4) Configuration of the code, including the hyper-parameters for PAHMC. 
	Refer to the manual for the shape and type requirements. Also note that a 
	lot of them can take either a single or an array/list of values. See user 
	manual for details.

It is suggested that the user keep a lookup table for the model paramters to 
make it easier to preserve order when working on the above steps.
