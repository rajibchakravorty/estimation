# -*- coding: utf-8 -*-

import numpy as np

from sensor_measurement import SensorMeasurement as SM
'''
    generates ground truth states of the object : 
        linear function of the immediate past state
'''

def generate2DGroundTruthState( XInit, N_iter, \
                              transitionMatrix, \
                              input ):
    
    ## building up the trajectory ground truth
    stateSize = XInit.shape[ 0 ]
    groundStates = np.zeros( ( stateSize, N_iter ) )
    
    groundStates[ :, 0 ]    = XInit
    for i in np.arange( 1, N_iter ):
        
        Xt = np.reshape(  groundStates[ :, i - 1 ], ( stateSize, 1 ) ) 
        
        Xt1 = np.dot( transitionMatrix, Xt ) + input
        
        groundStates[:,i] = np.reshape( Xt1, ( stateSize, ) )

    return groundStates
    
    
'''

    Function to generate measurements : linear function of the state
    
'''
def generateMeasuremnent( states, H, R, N_iter ):
    
    
    measurements = list()
    
    
    ##modelling the sensor returns
    measSize  = H.shape[ 0 ]
    stateSize = H.shape[ 1 ]
    for i in np.arange( 0, N_iter ):
        
        Xt = np.reshape(  states[ :, i ], ( stateSize, 1 ) ) 
        
        Y = np.dot( H, Xt ) + np.reshape( np.random.multivariate_normal( [0,0], R , 1 ), (measSize, 1 ) )
        
        meas = SM( True, measSize, 1, i )
        meas.measurements = Y
        
        measurements.append( meas )
        ##measurements[:,i ] = np.reshape( Y, (measSize, ) )
        
    return measurements



'''

    Function to generate measurements : linear function of the state
    Models missed detection and false detections
    
    True target detected with a probability of detection == detectionProb
    False measurements are generated with a Poisson distribution specifed
    by the density (falseMeasDensity )/scan/square unit of space and then
    spreaded accross the 2D world uniformly
    
'''
def generateMeasurements( states, H, R, \
                          detectionProb, falseMeasDensity, \
                          N_iter, worldSize, forceInitiation = True ):
    
    
    measurements = list()
    
    targetDetection    = np.random.rand( N_iter  )
    falseMeasurements  = np.random.poisson( falseMeasDensity * worldSize[0] * worldSize[1], N_iter)    
    
    
    ##modelling the sensor returns
    measSize  = H.shape[ 0 ]
    stateSize = H.shape[ 1 ]
    for i in np.arange( 0, N_iter ):
        
        if( ( forceInitiation == True ) and ( i < 2) ):
            meas = SM( True, measSize, 1, i )
            Xt = np.reshape(  states[ :, i ], ( stateSize, 1 ) ) 
            Y = np.dot( H, Xt ) + np.reshape( np.random.multivariate_normal( [0,0], R , 1 ), (measSize, 1 ) )
        
            meas.measurements = Y
        
        else:
            
            world = np.tile( np.reshape( worldSize, ( 2, 1 ) ) , (1,falseMeasurements[ i ] ) )
                
            if( targetDetection[ i ] <= detectionProb ):
                
                measurementCount = falseMeasurements[ i ] + 1
                meas = SM( True, measSize, measurementCount, i )
                Xt = np.reshape(  states[ :, i ], ( stateSize, 1 ) ) 
                Y = np.dot( H, Xt ) + np.reshape( np.random.multivariate_normal( [0,0], R , 1 ), (measSize, 1 ) )
        
                
                falseXY = np.random.rand( 2, falseMeasurements[ i ] ) * world
                meas.measurements = np.concatenate( ( Y, falseXY ), axis = 1 )
                
            else:
                
                measurementCount = falseMeasurements[ i ]
                meas = SM( False, measSize, measurementCount, i )
                
                falseXY = np.random.rand( 2, falseMeasurements[ i ] ) * world
                meas.measurements = falseXY
                            
                
        
        measurements.append( meas )
        ##measurements[:,i ] = np.reshape( Y, (measSize, ) )
        
    return measurements
