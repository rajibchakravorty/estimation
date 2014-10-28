# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:39:51 2014

@author: rchakrav
"""


"""
    Runs KF 1000 times; collected the estimated states (position, velocity);
    calculate the error in each run and displays the Mean Squared Error of
    states in 1000 runs.
    
    The target state transition matrix and the sensor model- both are linear.
    
    
    On each run, 
    
    ** the target appears at [100.0, 200.0, 35, 0 ] 
    (x_pos, y_pos, x_vel, y_vel ).
    
    ** the measurements are generated
    


"""




import numpy as np
from kalmanfilter.kalman_filter import KalmanFilter
from utility.utility import generate2DGroundTruthState, generateMeasuremnent


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
    
    
    monte_carlo_repeat = 1000
    
    ##----------- Declaring variables and constants
    
    
    stateSize = F.shape[0]
    measSize  = H.shape[0]
    
    squaredError = np.zeros( ( stateSize, N_iter ) )

    ## building up the trajectory ground truth
    groundTruthStates = generate2DGroundTruthState( XInit, N_iter, F, np.dot( B, U ) )      

    ##------------------ Kalman fitler starts

    ## KF parameters are independent of Monte Carlo Run
    ## and hence KF can be initiated here

    ## initiation
    kf = KalmanFilter( )
    
    ## initiate the model parameters
    kf.initModel( F, H, Q, R, B, U )



    ## repeat the algorithm for a number of 
    ## times and collect the error
    ## each time the random num gen produces a different set
    ## of errors and at the end the average error will
    ## indicate how the algorithm performs
    
    for mcr in np.arange( 0, monte_carlo_repeat ):
        
        ##generate the measurements
        measurements      = generateMeasuremnent( groundTruthStates, H, R, N_iter )
    
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
            
            
            ## for monte carlo run we take the advantage of runFilter method:
            ## of KF
            X, estimatedCov = kf.runFilter( Xt, estimatedCov, Yt )
            
            estimatedStates[ :, i ] = np.reshape( X, (stateSize, ) )


        ##after one run of the KF for all the time steps, time to collect the
        #squared error
         
        squaredError = squaredError + ( groundTruthStates - estimatedStates ) ** 2

    
    rmse = np.sqrt( squaredError / monte_carlo_repeat )
    
    
    ## some plotting function to see how we have fared
    time_axis = dt * np.arange( 2, N_iter )
    plt.plot( time_axis, rmse[0,2: ] )
    plt.ylim( [0, max( rmse[0,2: ] ) ] )
    plt.xlabel( 'Time' )
    plt.ylabel( 'Mean Squared Error (X-Pos)' )    
    plt.show()


    plt.plot( time_axis, rmse[1,2: ] )
    plt.xlabel( 'Time' )
    plt.ylim( [0, max( rmse[1,2: ] ) ] )
    plt.ylabel( 'Mean Squared Error (Y-Pos)' )    
    plt.show()
    
    plt.plot( time_axis, rmse[2,2: ] )
    plt.xlabel( 'Time' )
    plt.ylim( [0, max( rmse[2,2: ] ) ] )
    plt.ylabel( 'Mean Squared Error (X-Velocity)' )    
    plt.show()
    
    plt.plot( time_axis, rmse[3,2: ] )
    plt.xlabel( 'Time' )
    plt.ylim( [0, max( rmse[3,2: ] ) ] )
    plt.ylabel( 'Mean Squared Error (Y-Velocity)' )    
    plt.show()