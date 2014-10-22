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
    
    def runFilter( self, x0, P0 ):
        
        x10, P10 = self.predict( x0, P0 )
        yhat, S  = self.predictedMeas( x10, P10 )         
        
        '''
            TODO : gating, association, likelihood calculation, mixture update
            
        '''
        
    def predict( self, x0, P0 ):
        
        x10, p10 = self.kf.predict( x0, P0 )
    
    def predictedMeas( self, x10, P10 ):
        
        return self.kf.predictedMeas( x10, p10 )
        
    def innov( self, y1, x10, P10 ):
        
        ytilde = y1 - dot( self.H, x10 )
        
        S = dot( self.H, dot( P10, self.H.T ) ) + self.R
        
        return ytilde, S
    
    def update( self, x10, P10, ytilde, S ):
        
        K   = dot( P10, dot( self.H.T, inv( S ) ) )
        x11 = x10 + dot( K, ytilde )
        P11 = P10 - dot( K, dot( self.H, P10 ) )

        return x11, P11
    