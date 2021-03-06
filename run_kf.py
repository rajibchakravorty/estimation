# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:39:51 2014

@author: rchakrav
"""

import numpy as np
from kalmanfilter.kalman_filter import KalmanFilter
from utility.utility import generate2DGroundTruthState, generateMeasuremnent
from utility.sensor_measurement import SensorMeasurement as SM

import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    
    
    ##----------- Declaring variables and constants
        
    #time step of mobile movement
    dt = 1.0
    
    # Initialization of state matrices
    
    ## initial state of the object on a 2d plane
    ## starts at position (0,0) with velocity (0.1, 0.1)

    XInit = np.array( [100.0, 200.0, 35.0, 0.0 ] )             
    
    
    ## state transition matrix
    ##the following matrix assumes a constant velocity 
    ##in both of the axes
    F = np.array([ [ 1.0,0,dt,0 ],\
                  [0,1.0,0,dt],\
                  [ 0, 0,1.0,0 ],\
                  [0,0,0,1.0] ])
    
    ## covariance to capture the uncertainty in the state transition
    Q = 2.0 * np.array( [ [0.25 * np.power( dt, 4), 0, 0.5 * np.power( dt , 3), 0], \
                 [0, 0.25 * np.power( dt, 4), 0  , 0.5 * np.power( dt , 3)],\
                 [0.5 * np.power( dt , 3), 0, np.power( dt, 2 ), 0 ], 
                 [0, 0.5 * np.power( dt , 3), 0, np.power( dt, 2 ) ] ] )
    Q = np.reshape( Q, ( XInit.shape[0], XInit.shape[0] ) )   

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
    R = 25 * np.eye( H.shape[0] )

    # Number of iterations/time steps
    N_iter = 30
    
    ##----------- Declaring variables and constants
    
    
    stateSize = F.shape[0]
    measSize  = H.shape[0]

    ## building up the trajectory ground truth
    groundTruthStates = generate2DGroundTruthState( XInit, N_iter, F, np.dot( B, U ) )      

    ##generate the measurements
    measurements      = generateMeasuremnent( groundTruthStates, H, R, N_iter )
    


    ##------------------ Kalman fitler starts


    ## initiation
    kf = KalmanFilter( )
    
    
    ## initiate the model parameters
    kf.initModel( F, H, Q, R, B, U )
        
    
    
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
    for i in np.arange(2, N_iter):
        
        Xt = np.reshape( estimatedStates[ :, i - 1 ], ( stateSize, 1 ) ) 
        Yt = np.reshape( measurements[ i ].measurements, ( measSize, 1 ) )
        
        X, P = kf.predict( Xt, estimatedCov )
        yhat, S = kf.predictedMeas( X, P )
        
        ytilde = kf.innov( Yt, yhat )
        
        X, estimatedCov = kf.update( X, P, ytilde, S )
        
        estimatedStates[ :, i ] = np.reshape( X, (stateSize, ) )
        

    ## some plotting function to see how we have fared
    plt.plot( groundTruthStates[0,2: ], groundTruthStates[1,2: ] )
    plt.plot( estimatedStates[0,2: ], estimatedStates[1,2: ] )
    ##plt.plot( measurements[0,1:], measurements[1,1:]  )
    plt.legend( ['True Trajectory','Estimated Trajectory'], loc = 'upper left' )
    plt.xlabel( 'X' )
    plt.ylabel( 'Y' )
    plt.show()
    
    ##plt.plot( XS[0,: ] )
    ##plt.plot( XS[0,: ] )
    
                