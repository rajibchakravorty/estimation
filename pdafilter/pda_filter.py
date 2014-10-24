# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:11:13 2014

@author: rchakrav
"""

import sys
sys.path.insert( 0, '../kalmanfilter')

import numpy as np
import numpy.linalg as lin

from numpy import dot

from kalmanfilter.kalman_filter import KalmanFilter


class PDAFilter( object ):
    
    
    def initModel( self, F, H, Q, R, B, U, probDetection = 0.9, lam = 0 ):
        
        self.kf = KalmanFilter()
    
        self.kf.initModel( F, H, Q, R, B, U )
    
        self.PD = probDetection
        
        self.lam = lam
    

    def predict( self, x0, P0 ):
        
        return self.kf.predict( x0, P0 )
        
        

    def predictedMeas( self, x10, P10 ):
        
        return self.kf.predictedMeas( x10, P10 )
        
        

    def innov( self, yi, yhat ):
        
        return self.kf.innov( yi, yhat )
        
    
    def updateWithMeas( self, x10, P10, ytilde, S ) :
        
        return self.kf.update( x10, P10, ytilde, S)
    
    
    def runFilter( self, x0, p0, ys, gateProbability, gateVolume ):
        
        
        stateSize = x0.shape[0]
        measSize  = ys.shape[0]
                
        x10, p10 = self.predict( x0, p0 )
        
        yhat, S = self.predictedMeas( x10, p10 )
        
        sInv = lin.inv( S )
        
        
        totalReturns = ys.shape[1]
        x11s = np.zeros( (stateSize, totalReturns + 1 ) )
        p11s = np.zeros( ( stateSize, stateSize, totalReturns+ 1 ) )
        betas = np.zeros( totalReturns + 1 )
        
        x11s[ :, 0 ]    = np.reshape( x10, (stateSize ) )
        p11s[ :, :, 0 ] = np.reshape( p10, ( stateSize, stateSize ) )
        
        betas[ 0 ] = gateProbability * np.sqrt( lin.det( 2.0 * np.pi * S  ) ) / gateVolume

        
        for m in np.arange( 0, totalReturns ):
            
            Yt = np.reshape( ys[:,m], ( measSize, 1 ) )
            
            ytilde = self.innov( Yt, yhat )

            
            x11, p11 = self.updateWithMeas( x10, p10, ytilde, S )
            
            x11s[ :, m+1 ]   = np.reshape( x11, ( stateSize, ) )
            p11s[:, :, m+1 ] = np.reshape( p11, (stateSize , stateSize ) )
            
            betas[ m ] = np.exp( -0.5 * dot( ytilde.T, dot( sInv, ytilde  ) ) )
            
            
        return x11s, p11s, betas
        
        
            
            