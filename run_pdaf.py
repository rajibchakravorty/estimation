# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:39:51 2014

@author: rchakrav
"""

import numpy as np
from kalmanfilter.kalman_filter import KalmanFilter
from utility.utility import generate2DGroundTruthState, generateMeasurements


import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    
    
    ##----------- Declaring variables and constants
        
    #time step of mobile movement
    dt = 0.1
    
    # Initialization of state matrices
    
    ## initial state of the object on a 2d plane
    ## starts at position (0,0) with velocity (0.1, 0.1)

    XInit = np.array( [0.0, 0.0, 1.0, 1.0 ] )             
    
    
    ## state transition matrix
    ##the following matrix assumes a constant velocity 
    ##in both of the axes
    F = np.array( [ [1, 0, dt , 0], \
                 [0, 1, 0  , dt],\
                 [0, 0, 1, 0 ], 
                 [0, 0, 0,1 ] ] )
                 
    
    ##covariance to capture the uncertainty in the state transition
    Q = np.eye( XInit.shape[0] )

    ## models acceleration in the state transition
    B = np.eye( XInit.shape[0] )
    U = np.zeros( (XInit.shape[0], 1 ) )
    
    ## model the sensor:
    ##linear function of the state
    ##the following H models sensor that measures the (x,y) position of
    ##the object in a 2D plane
    H = np.array( [ [1, 0, 0, 0], \
                    [0, 1, 0, 0] ])
    
    
    #covariance of the sensor mesuarement error
    R = 0.01 * np.eye( H.shape[0] )

    # Number of iterations/time steps
    N_iter = 50
    
    # detection probability
    PD = 0.9
    
    ## density of false measuremetns /scan/m^2
    lam = 1e-4
    
    ## 2D world size
    worldSize = np.array( [1000, 400] )
    
    stateSize = F.shape[0]
    measSize  = H.shape[0]

    ## building up the trajectory ground truth
    groundTruthStates = generate2DGroundTruthState( XInit, N_iter, F, np.dot( B, U ) )      

    ##generate the measurements
    measurements      = generateMeasurements( groundTruthStates, H, R, \
                                              PD, lam, \
                                              N_iter, worldSize )
    
    