# pahmc-ode-cpu

A user-manual is forthcoming. Meanwhile, the docstrings within the code should be self-explanatory enough to get you started.

Requirements:
1) Python 3.7 or above;
2) Numpy 1.16 or above;
3) Numba 0.45 or above;
4) Matplotlib 3.1 or above;
5) (Optional) PyTorch 1.2 or above.

Before running the code, the user is assumed to have the following:
1) A dynamical system. If calling one of the built-in models, the name of the dynamics must have a match in "lib_dynamics.py"; if builing from scratch, 'def_dynamics.py' must be ready to go.

2) The data. If performing twin-experiments, the specs should be given but a data file is not required; if working with real data, the data should be prepared according to the user manual.

3) If external stimuli are needed, a .npy file containing the time series; 

4) Configuration of the code, including the hyper-parameters for PAHMC. Refer to the manual for the shape and type requirements. Also note that a lot of them can take either a single number or an array. See user manual for details.

It is suggested that the user keep a lookup table for the paramters in the model to make it easier to remember the order.
