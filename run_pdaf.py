# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:39:51 2014

@author: rchakrav
"""

import numpy as np
from pdafilter.pda_filter import PDAFilter
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
    PD = 1.0
    
    ## density of false measuremetns /scan/m^2
    lam = 0
    
    ## 2D world size
    worldSize = np.array( [10, 10] )
    
    stateSize = F.shape[0]
    measSize  = H.shape[0]

    ## building up the trajectory ground truth
    groundTruthStates = generate2DGroundTruthState( XInit, N_iter, F, np.dot( B, U ) )      

    ##generate the measurements
    measurements      = generateMeasurements( groundTruthStates, H, R, \
                                              PD, lam, \
                                              N_iter, worldSize )
    
    ##------------------ Kalman fitler starts


    ## initiation
    pdaf = PDAFilter( )
    
    
    ## initiate the model parameters
    pdaf.initModel( F, H, Q, R, B, U, PD, lam )
        
    
    
    # the filter loop
    estimatedStates = np.zeros( ( F.shape[0], N_iter ) )
    
    
    ## --- initiating the filter with first estimate
    ## initiates the state and the covariance with
    ## 2-point initiation method
    
    y1 = np.reshape( measurements[ 1 ].measurements, ( measSize, 1 ) )
    y0 = np.reshape( measurements[ 0 ].measurements, ( measSize, 1 ) )
    
    estimatedStates[ :, 1 ] = [ y1[0], y1[1],\
                                (1/dt) * (y1[0] - y0[0]),\
                                (1/dt) * (y1[1] - y0[1]) ]
                                
    ## initial gues of the covariance with fixed values
    estimatedCov = np.concatenate( ( np.concatenate( ( R, (1.0/dt) * R ), axis = 1 ),\
                                     np.concatenate( ( (1.0/dt) * R, (1.0/dt)*(1.0/dt)*(R+R) ), axis = 1 ) ), axis = 0 )
    
    for i in np.arange( 2, N_iter ):
        
        '''
            TODO : PDA filter loop, gating/validation, Gaussian mixture
        '''
        
        ## collect the measurement
        
        ## ideally valid measurements should be gated
        ## for now validate everything
        
        sensorReturns = measurements[ i ].measurements
        
        Xt = np.reshape( estimatedStates[ :, i - 1 ], ( stateSize, 1 ) )
        
        x11s, P11s,betas = pdaf.runFilter( Xt, estimatedCov, sensorReturns, 1, 25 )
        
        X, estimatedCov = pdaf.mixOutput( x11s, P11s, betas )
        
        estimatedStates[ :, i ] = np.reshape( X, (stateSize, ) )
        
        
    ## some plotting function to see how we have fared
    plt.plot( groundTruthStates[0,2: ], groundTruthStates[1,2: ] )
    plt.plot( estimatedStates[0,2: ], estimatedStates[1,2: ] )
    ##plt.plot( measurements[0,1:], measurements[1,1:]  )
    plt.legend( ['True Trajectory','Estimated Trajectory'], loc = 'upper left' )
    plt.xlabel( 'X' )
    plt.ylabel( 'Y' )
    plt.show()
        
        
        