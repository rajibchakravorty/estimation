# -*- coding: utf-8 -*-

import numpy as np

import numpy.linalg as lin


def guassmix( means, covariances, weights ):
    
    noOfGaussian = means.shape[ 1 ]
    
    meanVectorSize = means.shape[ 0 ]
    
    
    mixedMean = np.zeros( ( meanVectorSize, 1 ) )
    mixedCov  = np.zeros( ( meanVectorSize, meanVectorSize ) )

    for g in np.arange( 0, noOfGaussian ):
        
        x11 = np.reshape( means[:,g], ( meanVectorSize, 1) )
        mixedMean = mixedMean + x11 * weights[g]
        
        
    for g in np.arange( 0, noOfGaussian ):
        
        x11 = np.reshape( means[:,g], ( meanVectorSize, 1) )
        p11 = np.reshape( covariances[ :, :, g ], ( meanVectorSize, meanVectorSize ) )
        mixedCov = mixedCov + weights[g] * ( p11 + np.dot( ( x11-mixedMean ), ( x11-mixedMean ).T ) )
        
        
    return mixedMean, mixedCov
    
    
def validateReturns( measurements, yhat, S,  validationWindow ):
    
    totalMeasurements = measurements.shape [ 1 ]
    measVector = measurements.shape[0]    
    
    sInv = lin.inv( S )    

    selectedReturns =  np.empty( ( measVector, 1 ) )
    for m in np.arange( 0, totalMeasurements ):
        
        ym = np.reshape( measurements[ :, m], ( measVector, 1  ) )
        
        stat = np.dot( ( ym - yhat ).T, np.dot( sInv, ( ym - yhat ) ) )
        
        if( stat < validationWindow ):
            
            selectedReturns = np.concatenate( (selectedReturns, ym ), axis = 1 )
            
    ##the first column of selected returns is a garbage; created when the array
    ##was initialized with np.empty(). so that has to be discarded
    
    if( selectedReturns.shape[1] == 1 ):
        selectedReturns = None
    else:
        selectedReturns = selectedReturns[:,1: ]
            
    gateVolume = validationWindow * np.pi * np.sqrt( lin.det( S ) )
        
    return selectedReturns, gateVolume
        
        