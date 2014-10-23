# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:11:13 2014

@author: rchakrav
"""

import sys
sys.path.insert( 0, '../kalmanfilter')

from kalmanfilter.kalman_filter import KalmanFilter


from numpy import dot, sum, tile, linalg
from numpy.linalg import inv


class PDAFilter( object ):
    
    self.kf = KalmanFilter()
    
    def initModel( self, F, H, Q, R, B, U, probDetection = 0.9, lam ):
        
        self.kf.initModel( F, H, Q, R, B, U )
    
        self.PD = probDetection
        
        self.lam = lam
    
    def runFilter( self, x0, P0, ys ):

        
        '''
            TODO : gating, association, likelihood calculation, mixture update
            
        '''
        
        if( ys.shape[1] == 0 ):
            
            self.kf.predict( x0, P0 )
            
        
        totalValidatedMeas = ys.shape[1]


        stateSize = x0.shape[0]
        measSize  = y0.shape[0]
        x11s = np.zeros( ( stateSize, totalValidation + 1 ) )
        p11s = np.zeros( ( stateSize, stateSize, totalValidation + 1 ) )
        x11s[ :,0 ] = np.reshape( x10, ( stateSize, 1 ) ) 
        p11s[ :,:,0 ] = np.reshape( p10, ( stateSize, stateSize ) )
        
        for vm in arange( 1, totalValidatedMeas ):
            
            yt = np.reshape( ys[:,vm], ( measSize, 1 ) )            
            x11u, p11u = self.kf.runFilter( x10, P10, yt ):
                
            x11s[ :,0 ] = np.reshape( x11u, ( stateSize, 1 ) ) 
            p11s[ :,:,0 ] = np.reshape( p11u, ( stateSize, stateSize ) )
            
            
        
        
    
    def update( self, x10, P10, ytilde, S ):
        
        K   = dot( P10, dot( self.H.T, inv( S ) ) )
        x11 = x10 + dot( K, ytilde )
        P11 = P10 - dot( K, dot( self.H, P10 ) )

        return x11, P11
    