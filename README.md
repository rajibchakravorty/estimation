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

Kalman Filter:
--------------

Assumes:

- f() and h() are linear
- w(k), v(k) are additive
- w(k), v(k) are both normally distributed with mean 0 and variance Q and R
respectively
- w(i) and w(j) are uncorrelated, i \neq j
- v(i) and v(j) are uncorrelated, i \neq j
- measurement from target is available at each scan
- no false measurements generated