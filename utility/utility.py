# -*- coding: utf-8 -*-


'''
    generates ground truth states of the object : 
        linear function of the immediate past state
'''

def generateGroundTruthState( XInit, N_iter, \
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
def generateMeasuremnent( states, H, R ):
    
    ##modelling the sensor returns
    measSize  = H.shape[ 0 ]
    stateSize = H.shape[ 1 ]
    measurements = np.zeros( ( measSize, N_iter ) )
    for i in np.arange( 0, N_iter ):
        
        Xt = np.reshape(  states[ :, i ], ( stateSize, 1 ) ) 
        
        Y = np.dot( H, Xt ) + np.reshape( np.random.multivariate_normal( [0,0], R , 1 ), (measSize, 1 ) )
        
        measurements[:,i ] = np.reshape( Y, (measSize, ) )
        
    return measurements
