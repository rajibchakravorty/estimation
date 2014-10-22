Basics of Estimation Algorithm
====================

A target moves with a state x(k). The transition of states is give by:

x(k) = f( x(k-1) , w(k) ) ......(1)

In general w(k) is a random perturbation and f() is non-linear.

A sensor measures some aspect of the target state. This is given by

y(k) = h( x(k), v(k) ) ...... (2)

In general, v(k) is a random error in the measurement and h() is non-linear.

The problem is to estimate the state x(k) given a sequence of measurements.

Each estimation algorithm makes an assumption of these basic models and/or
adds complexities on top of the basic model (adding more assumption and model
parameter)

The project is aimed to implement all the major, if not all, estimation/tracking
algorithms in Python.

The objective of the project is to provide a Python based free platform (as opposed 
to Matlab, which is the primary platform used for this field of research) for 
gradutate students and researchers to conduct experiments. Besides this core objective, 
other advantages include:

1) Possible speed-up through parallelization (easier in Python)
2) Develop standardized test cases and peer-review of codes

Use of the software is under GPL-3.0 license agreement.

Dev environment
===============

Anaconda (http://continuum.io/downloads)
