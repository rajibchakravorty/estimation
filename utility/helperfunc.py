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